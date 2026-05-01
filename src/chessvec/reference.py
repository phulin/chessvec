"""Reference (single-state, scalar) chess rules engine.

The goal of this module is correctness and readability. It is the spec the
vectorized engine is tested against. It implements the full rules of chess
except threefold repetition (which is deferred); the 50-move rule and
insufficient-material draws are included.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .action_encoding import encode_move, file_of, rank_of, square
from .types import (
    ACTION_SIZE,
    BLACK,
    BLACK_WIN,
    CR_BK,
    CR_BQ,
    CR_WK,
    CR_WQ,
    DRAW,
    ONGOING,
    PIECE_CHARS,
    WHITE,
    WHITE_WIN,
    Piece,
    char_piece,
    piece_color,
    piece_type,
)

# Move tuples: (from_sq, to_sq, promotion). promotion=0 if none, else 2..5
# for piece-type N,B,R,Q.

Move = tuple[int, int, int]

# ----- Geometry helpers ----------------------------------------------------

KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
KING_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (df, dr) for df in (-1, 0, 1) for dr in (-1, 0, 1) if (df, dr) != (0, 0)
)
ROOK_DIRS: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS: tuple[tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))
QUEEN_DIRS_ALL: tuple[tuple[int, int], ...] = ROOK_DIRS + BISHOP_DIRS


def in_board(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


# ----- State ---------------------------------------------------------------


@dataclass
class State:
    board: list[int] = field(default_factory=lambda: [0] * 64)
    side_to_move: int = WHITE
    castling: int = 0b1111  # KQkq
    en_passant: int = -1
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def copy(self) -> State:
        return replace(self, board=list(self.board))

    # ---- string I/O (used by tests; FEN-compatible) ----

    @classmethod
    def starting(cls) -> State:
        return cls.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    @classmethod
    def from_fen(cls, fen: str) -> State:
        parts = fen.split()
        if len(parts) != 6:
            raise ValueError(f"FEN must have 6 fields: {fen!r}")
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ValueError("FEN board must have 8 rows")
        board = [0] * 64
        for fen_rank, row in enumerate(rows):
            rank = 7 - fen_rank
            f = 0
            for c in row:
                if c.isdigit():
                    f += int(c)
                else:
                    board[square(f, rank)] = char_piece(c)
                    f += 1
            if f != 8:
                raise ValueError(f"FEN row has wrong width: {row!r}")
        side = WHITE if parts[1] == "w" else BLACK
        cr = 0
        if "K" in parts[2]:
            cr |= 1 << CR_WK
        if "Q" in parts[2]:
            cr |= 1 << CR_WQ
        if "k" in parts[2]:
            cr |= 1 << CR_BK
        if "q" in parts[2]:
            cr |= 1 << CR_BQ
        ep = -1
        if parts[3] != "-":
            ep_file = ord(parts[3][0]) - ord("a")
            ep_rank = int(parts[3][1]) - 1
            ep = square(ep_file, ep_rank)
        return cls(
            board=board,
            side_to_move=side,
            castling=cr,
            en_passant=ep,
            halfmove_clock=int(parts[4]),
            fullmove_number=int(parts[5]),
        )

    def to_fen(self) -> str:
        rows: list[str] = []
        for fen_rank in range(8):
            rank = 7 - fen_rank
            row = ""
            empty = 0
            for f in range(8):
                p = self.board[square(f, rank)]
                if p == 0:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += PIECE_CHARS[p]
            if empty:
                row += str(empty)
            rows.append(row)
        side = "w" if self.side_to_move == WHITE else "b"
        cr = ""
        if self.castling & (1 << CR_WK):
            cr += "K"
        if self.castling & (1 << CR_WQ):
            cr += "Q"
        if self.castling & (1 << CR_BK):
            cr += "k"
        if self.castling & (1 << CR_BQ):
            cr += "q"
        cr = cr or "-"
        ep = self._ep_fen_field()
        return f"{'/'.join(rows)} {side} {cr} {ep} {self.halfmove_clock} {self.fullmove_number}"

    def _ep_fen_field(self) -> str:
        """Match python-chess's default behavior: only emit the en-passant
        square if the side-to-move actually has a pawn able to capture
        there."""
        if self.en_passant < 0:
            return "-"
        ep = self.en_passant
        # Pawns of side-to-move that could capture on `ep` are on the rank
        # one square "behind" ep from their POV (i.e. same rank as the pushed
        # pawn), on adjacent files.
        my_pawn = Piece.WP.value if self.side_to_move == WHITE else Piece.BP.value
        ep_f = file_of(ep)
        ep_r = rank_of(ep)
        # The capturing pawn is at rank (ep_r - 1) for white, (ep_r + 1) for black,
        # which is the same as the rank of the just-pushed pawn:
        capturing_rank = ep_r - 1 if self.side_to_move == WHITE else ep_r + 1
        for df in (-1, 1):
            f = ep_f + df
            if 0 <= f < 8 and self.board[capturing_rank * 8 + f] == my_pawn:
                return chr(ord("a") + ep_f) + str(ep_r + 1)
        return "-"


# ----- Attack detection ----------------------------------------------------


def square_attacked(board: list[int], sq: int, by_color: int) -> bool:
    """Is `sq` attacked by any piece of `by_color`?"""
    f0, r0 = file_of(sq), rank_of(sq)

    # Pawn attacks: a pawn of `by_color` attacks `sq` if there's a same-color
    # pawn one rank "behind" `sq` (relative to by_color's forward direction)
    # on an adjacent file.
    pawn_dir = 1 if by_color == WHITE else -1
    pawn = Piece.WP if by_color == WHITE else Piece.BP
    for df in (-1, 1):
        f, r = f0 + df, r0 - pawn_dir
        if in_board(f, r) and board[square(f, r)] == pawn:
            return True

    # Knight.
    knight = Piece.WN if by_color == WHITE else Piece.BN
    for df, dr in KNIGHT_OFFSETS:
        f, r = f0 + df, r0 + dr
        if in_board(f, r) and board[square(f, r)] == knight:
            return True

    # King.
    king = Piece.WK if by_color == WHITE else Piece.BK
    for df, dr in KING_OFFSETS:
        f, r = f0 + df, r0 + dr
        if in_board(f, r) and board[square(f, r)] == king:
            return True

    # Sliders. Walk each direction; first non-empty square decides.
    rook = Piece.WR if by_color == WHITE else Piece.BR
    bishop = Piece.WB if by_color == WHITE else Piece.BB
    queen = Piece.WQ if by_color == WHITE else Piece.BQ
    for df, dr in ROOK_DIRS:
        f, r = f0 + df, r0 + dr
        while in_board(f, r):
            p = board[square(f, r)]
            if p:
                if p == rook or p == queen:
                    return True
                break
            f += df
            r += dr
    for df, dr in BISHOP_DIRS:
        f, r = f0 + df, r0 + dr
        while in_board(f, r):
            p = board[square(f, r)]
            if p:
                if p == bishop or p == queen:
                    return True
                break
            f += df
            r += dr

    return False


def find_king(board: list[int], color: int) -> int:
    target = Piece.WK if color == WHITE else Piece.BK
    for sq, p in enumerate(board):
        if p == target:
            return sq
    return -1


def in_check(state: State, color: int) -> bool:
    ksq = find_king(state.board, color)
    if ksq < 0:
        return False
    return square_attacked(state.board, ksq, 1 - color)


# ----- Pseudo-legal move generation ----------------------------------------


def _gen_pawn_moves(state: State, from_sq: int, out: list[Move]) -> None:
    me = state.side_to_move
    forward = 1 if me == WHITE else -1
    start_rank = 1 if me == WHITE else 6
    promo_rank = 7 if me == WHITE else 0
    f0, r0 = file_of(from_sq), rank_of(from_sq)

    # Single push.
    r1 = r0 + forward
    if in_board(f0, r1) and state.board[square(f0, r1)] == 0:
        if r1 == promo_rank:
            for promo in (2, 3, 4, 5):
                out.append((from_sq, square(f0, r1), promo))
        else:
            out.append((from_sq, square(f0, r1), 0))
            # Double push.
            r2 = r0 + 2 * forward
            if r0 == start_rank and state.board[square(f0, r2)] == 0:
                out.append((from_sq, square(f0, r2), 0))

    # Captures.
    for df in (-1, 1):
        f1 = f0 + df
        if not in_board(f1, r1):
            continue
        target_sq = square(f1, r1)
        target = state.board[target_sq]
        if target != 0 and piece_color(target) != me:
            if r1 == promo_rank:
                for promo in (2, 3, 4, 5):
                    out.append((from_sq, target_sq, promo))
            else:
                out.append((from_sq, target_sq, 0))
        elif target == 0 and target_sq == state.en_passant:
            out.append((from_sq, target_sq, 0))


def _gen_jump_moves(
    state: State, from_sq: int, offsets: tuple[tuple[int, int], ...], out: list[Move]
) -> None:
    me = state.side_to_move
    f0, r0 = file_of(from_sq), rank_of(from_sq)
    for df, dr in offsets:
        f, r = f0 + df, r0 + dr
        if not in_board(f, r):
            continue
        target = state.board[square(f, r)]
        if target == 0 or piece_color(target) != me:
            out.append((from_sq, square(f, r), 0))


def _gen_slider_moves(
    state: State, from_sq: int, dirs: tuple[tuple[int, int], ...], out: list[Move]
) -> None:
    me = state.side_to_move
    f0, r0 = file_of(from_sq), rank_of(from_sq)
    for df, dr in dirs:
        f, r = f0 + df, r0 + dr
        while in_board(f, r):
            target = state.board[square(f, r)]
            if target == 0:
                out.append((from_sq, square(f, r), 0))
            else:
                if piece_color(target) != me:
                    out.append((from_sq, square(f, r), 0))
                break
            f += df
            r += dr


def _gen_castling(state: State, out: list[Move]) -> None:
    me = state.side_to_move
    board = state.board
    them = 1 - me
    if me == WHITE:
        rank = 0
        ksq = square(4, rank)
        if board[ksq] != Piece.WK:
            return
        if state.castling & (1 << CR_WK):
            if (
                board[square(5, rank)] == 0
                and board[square(6, rank)] == 0
                and board[square(7, rank)] == Piece.WR
                and not square_attacked(board, ksq, them)
                and not square_attacked(board, square(5, rank), them)
                and not square_attacked(board, square(6, rank), them)
            ):
                out.append((ksq, square(6, rank), 0))
        if state.castling & (1 << CR_WQ):
            if (
                board[square(3, rank)] == 0
                and board[square(2, rank)] == 0
                and board[square(1, rank)] == 0
                and board[square(0, rank)] == Piece.WR
                and not square_attacked(board, ksq, them)
                and not square_attacked(board, square(3, rank), them)
                and not square_attacked(board, square(2, rank), them)
            ):
                out.append((ksq, square(2, rank), 0))
    else:
        rank = 7
        ksq = square(4, rank)
        if board[ksq] != Piece.BK:
            return
        if state.castling & (1 << CR_BK):
            if (
                board[square(5, rank)] == 0
                and board[square(6, rank)] == 0
                and board[square(7, rank)] == Piece.BR
                and not square_attacked(board, ksq, them)
                and not square_attacked(board, square(5, rank), them)
                and not square_attacked(board, square(6, rank), them)
            ):
                out.append((ksq, square(6, rank), 0))
        if state.castling & (1 << CR_BQ):
            if (
                board[square(3, rank)] == 0
                and board[square(2, rank)] == 0
                and board[square(1, rank)] == 0
                and board[square(0, rank)] == Piece.BR
                and not square_attacked(board, ksq, them)
                and not square_attacked(board, square(3, rank), them)
                and not square_attacked(board, square(2, rank), them)
            ):
                out.append((ksq, square(2, rank), 0))


def pseudo_legal_moves(state: State) -> list[Move]:
    me = state.side_to_move
    out: list[Move] = []
    for sq in range(64):
        p = state.board[sq]
        if p == 0 or piece_color(p) != me:
            continue
        pt = piece_type(p)
        if pt == 1:
            _gen_pawn_moves(state, sq, out)
        elif pt == 2:
            _gen_jump_moves(state, sq, KNIGHT_OFFSETS, out)
        elif pt == 3:
            _gen_slider_moves(state, sq, BISHOP_DIRS, out)
        elif pt == 4:
            _gen_slider_moves(state, sq, ROOK_DIRS, out)
        elif pt == 5:
            _gen_slider_moves(state, sq, QUEEN_DIRS_ALL, out)
        elif pt == 6:
            _gen_jump_moves(state, sq, KING_OFFSETS, out)
    _gen_castling(state, out)
    return out


# ----- Make-move and legality filter ---------------------------------------


def apply_move(state: State, move: Move) -> State:
    """Return a new state after playing `move`. Assumes `move` is legal."""
    s = state.copy()
    from_sq, to_sq, promo = move
    moving = s.board[from_sq]
    captured = s.board[to_sq]
    pt = piece_type(moving)
    me = s.side_to_move

    # Reset en passant; we'll re-set it below if this is a double pawn push.
    new_ep = -1

    # En passant capture.
    if pt == 1 and to_sq == state.en_passant and captured == 0:
        ep_capture_sq = square(file_of(to_sq), rank_of(from_sq))
        s.board[ep_capture_sq] = 0
        captured = Piece.BP if me == WHITE else Piece.WP

    # Move the piece.
    s.board[to_sq] = moving
    s.board[from_sq] = 0

    # Promotion.
    if pt == 1 and promo:
        promo_piece = promo if me == WHITE else promo + 6
        s.board[to_sq] = promo_piece

    # Pawn double push: set en passant target square (the square *behind*
    # the pawn).
    if pt == 1 and abs(rank_of(to_sq) - rank_of(from_sq)) == 2:
        new_ep = square(file_of(from_sq), (rank_of(from_sq) + rank_of(to_sq)) // 2)

    # Castling: move the rook.
    if pt == 6 and abs(file_of(to_sq) - file_of(from_sq)) == 2:
        rank = rank_of(from_sq)
        if file_of(to_sq) == 6:  # kingside
            s.board[square(5, rank)] = s.board[square(7, rank)]
            s.board[square(7, rank)] = 0
        else:  # queenside
            s.board[square(3, rank)] = s.board[square(0, rank)]
            s.board[square(0, rank)] = 0

    # Update castling rights: remove rights when king moves, when own rook
    # moves from its original square, or when opponent's rook is captured on
    # its original square.
    cr = s.castling
    if moving == Piece.WK:
        cr &= ~((1 << CR_WK) | (1 << CR_WQ))
    if moving == Piece.BK:
        cr &= ~((1 << CR_BK) | (1 << CR_BQ))
    # Squares that, when departed-from or captured-on, drop a castling right.
    rook_squares = {
        square(0, 0): 1 << CR_WQ,
        square(7, 0): 1 << CR_WK,
        square(0, 7): 1 << CR_BQ,
        square(7, 7): 1 << CR_BK,
    }
    cr &= ~rook_squares.get(from_sq, 0)
    cr &= ~rook_squares.get(to_sq, 0)
    s.castling = cr

    # Halfmove clock.
    if pt == 1 or captured != 0:
        s.halfmove_clock = 0
    else:
        s.halfmove_clock = state.halfmove_clock + 1

    # Side and fullmove number.
    if me == BLACK:
        s.fullmove_number = state.fullmove_number + 1
    s.side_to_move = 1 - me
    s.en_passant = new_ep
    return s


def legal_moves(state: State) -> list[Move]:
    me = state.side_to_move
    out: list[Move] = []
    for mv in pseudo_legal_moves(state):
        nxt = apply_move(state, mv)
        if not in_check(nxt, me):
            out.append(mv)
    return out


# ----- Action mask & game result -------------------------------------------


def legal_action_mask(state: State) -> list[bool]:
    """Return a 4672-element list[bool] of legal actions in the AlphaZero
    move encoding. We resolve queen-promotion vs underpromotion by emitting
    only the queen-plane action for queen promotions (consistent with both
    engines)."""
    mask = [False] * ACTION_SIZE
    for from_sq, to_sq, promo in legal_moves(state):
        # Queen-promotion goes through the queen-like plane (promotion arg=0).
        # Underpromotions (N,B,R) go through the underpromotion planes.
        promo_arg = 0 if promo in (0, 5) else promo
        idx = encode_move(from_sq, to_sq, promo_arg, state.side_to_move)
        mask[idx] = True
    return mask


def insufficient_material(state: State) -> bool:
    """Per FIDE: K vs K, K+N vs K, K+B vs K, K+B vs K+B with same-color bishops."""
    pieces: list[tuple[int, int]] = []  # (piece, square)
    for sq, p in enumerate(state.board):
        if p == 0:
            continue
        pt = piece_type(p)
        if pt in (1, 4, 5):  # pawn, rook, queen -> sufficient
            return False
        pieces.append((p, sq))
    # Only kings, knights, bishops left.
    non_kings = [(p, sq) for p, sq in pieces if piece_type(p) != 6]
    if len(non_kings) == 0:
        return True
    if len(non_kings) == 1:
        return piece_type(non_kings[0][0]) in (2, 3)
    if len(non_kings) == 2:
        # Both bishops, same square color, opposite sides.
        if all(piece_type(p) == 3 for p, _ in non_kings):
            colors = {piece_color(p) for p, _ in non_kings}
            if len(colors) == 2:
                sq_colors = {(file_of(sq) + rank_of(sq)) & 1 for _, sq in non_kings}
                if len(sq_colors) == 1:
                    return True
    return False


def game_result(state: State) -> int:
    """Return ONGOING / WHITE_WIN / BLACK_WIN / DRAW."""
    if not legal_moves(state):
        if in_check(state, state.side_to_move):
            return BLACK_WIN if state.side_to_move == WHITE else WHITE_WIN
        return DRAW
    if state.halfmove_clock >= 100:  # 50-move rule (100 plies)
        return DRAW
    if insufficient_material(state):
        return DRAW
    return ONGOING
