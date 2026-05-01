"""Vectorized chess rules engine.

Operates on a batch of B environments simultaneously using PyTorch tensors.
The hot path contains no Python-level for loops over the batch; small fixed
loops over piece-types, ray-directions, and ray-distances (max 7) are used.

State tensors (all leading dim B):
    pieces            int8   [B, 64]  piece codes 0..12 (see types.PIECE_CHARS)
    side_to_move      int8   [B]      0 = white, 1 = black
    castling          int8   [B]      packed 4-bit KQkq mask (see types)
    en_passant        int8   [B]      square 0..63 or -1 (no ep)
    halfmove_clock    int16  [B]      plies since last pawn move/capture
    fullmove_number   int16  [B]      starts at 1, increments after black's move

Action space: 4672 = 64 * 73 (AlphaZero layout, see ``action_encoding.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .reference import State
from .types import (
    ACTION_SIZE,
    BLACK,
    BLACK_WIN,
    CR_BK,
    CR_BQ,
    CR_WK,
    CR_WQ,
    DRAW,
    NUM_MOVE_PLANES,
    NUM_PLANES,
    ONGOING,
    WHITE,
    WHITE_WIN,
    Piece,
)

# ---------------------------------------------------------------------------
# Direction helpers. Boards are reshaped to [..., 8, 8] with (rank, file)
# layout. shift_<dir> returns the board "as seen one square in <dir>" -- i.e.
# out[..., r, f] is in[..., r - dr, f - df].
# ---------------------------------------------------------------------------

# (df, dr) for each of the 8 queen-like directions, matching types.QUEEN_DIRS.
_QUEEN_SHIFTS: tuple[tuple[int, int], ...] = (
    (0, 1),  # N
    (1, 1),  # NE
    (1, 0),  # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1),  # SW
    (-1, 0),  # W
    (-1, 1),  # NW
)

# Knight (df, dr).
_KNIGHT_SHIFTS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)


def _shift(x: Tensor, df: int, dr: int) -> Tensor:
    """Shift a [..., 8, 8] tensor by (df, dr). Empty squares become False/0."""
    # For PyTorch F.pad on the last two dims, pad order is
    # (file_left, file_right, rank_bottom, rank_top).
    # Shifting (df>0) means out[r, f] = in[r, f - df]; trim files left, pad right.
    # Shifting (df<0) means out[r, f] = in[r, f - df]; trim files right, pad left.
    if df > 0:
        x = F.pad(x[..., :, :-df], (df, 0, 0, 0))
    elif df < 0:
        x = F.pad(x[..., :, -df:], (0, -df, 0, 0))
    if dr > 0:
        x = F.pad(x[..., :-dr, :], (0, 0, dr, 0))
    elif dr < 0:
        x = F.pad(x[..., -dr:, :], (0, 0, 0, -dr))
    return x


# ---------------------------------------------------------------------------
# Precomputed "from-square attacks" tables for jump pieces (knight, king,
# pawns). Shape [64, 64] bool: row=from-square, col=attacked square.
# ---------------------------------------------------------------------------


def _build_jump_table(offsets: tuple[tuple[int, int], ...]) -> Tensor:
    tbl = torch.zeros(64, 64, dtype=torch.bool)
    for from_r in range(8):
        for from_f in range(8):
            from_sq = from_r * 8 + from_f
            for df, dr in offsets:
                f, r = from_f + df, from_r + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    tbl[from_sq, r * 8 + f] = True
    return tbl


def _build_pawn_attack_table(forward: int) -> Tensor:
    tbl = torch.zeros(64, 64, dtype=torch.bool)
    for from_r in range(8):
        for from_f in range(8):
            from_sq = from_r * 8 + from_f
            for df in (-1, 1):
                f, r = from_f + df, from_r + forward
                if 0 <= f < 8 and 0 <= r < 8:
                    tbl[from_sq, r * 8 + f] = True
    return tbl


_KING_TABLE = _build_jump_table(
    tuple((df, dr) for df in (-1, 0, 1) for dr in (-1, 0, 1) if (df, dr) != (0, 0))
)
_KNIGHT_TABLE = _build_jump_table(_KNIGHT_SHIFTS)
_WPAWN_ATTACK = _build_pawn_attack_table(1)
_BPAWN_ATTACK = _build_pawn_attack_table(-1)


# ---------------------------------------------------------------------------
# Vectorized state container.
# ---------------------------------------------------------------------------


@dataclass
class VState:
    pieces: Tensor  # [B, 64] int8
    side_to_move: Tensor  # [B] int8
    castling: Tensor  # [B] int8
    en_passant: Tensor  # [B] int8 (-1 = none)
    halfmove_clock: Tensor  # [B] int16
    fullmove_number: Tensor  # [B] int16

    @property
    def batch_size(self) -> int:
        return int(self.pieces.shape[0])

    @property
    def device(self) -> torch.device:
        return self.pieces.device

    def clone(self) -> VState:
        return VState(
            pieces=self.pieces.clone(),
            side_to_move=self.side_to_move.clone(),
            castling=self.castling.clone(),
            en_passant=self.en_passant.clone(),
            halfmove_clock=self.halfmove_clock.clone(),
            fullmove_number=self.fullmove_number.clone(),
        )

    def to(self, device: torch.device | str) -> VState:
        return VState(
            pieces=self.pieces.to(device),
            side_to_move=self.side_to_move.to(device),
            castling=self.castling.to(device),
            en_passant=self.en_passant.to(device),
            halfmove_clock=self.halfmove_clock.to(device),
            fullmove_number=self.fullmove_number.to(device),
        )


def from_states(states: list[State], device: torch.device | str = "cpu") -> VState:
    B = len(states)
    pieces = torch.zeros(B, 64, dtype=torch.int8)
    side = torch.zeros(B, dtype=torch.int8)
    cr = torch.zeros(B, dtype=torch.int8)
    ep = torch.full((B,), -1, dtype=torch.int8)
    hmc = torch.zeros(B, dtype=torch.int16)
    fmn = torch.ones(B, dtype=torch.int16)
    for i, s in enumerate(states):
        for sq in range(64):
            pieces[i, sq] = s.board[sq]
        side[i] = s.side_to_move
        cr[i] = s.castling
        ep[i] = s.en_passant
        hmc[i] = s.halfmove_clock
        fmn[i] = s.fullmove_number
    vs = VState(pieces, side, cr, ep, hmc, fmn).to(device)
    return vs


def to_states(vs: VState) -> list[State]:
    out: list[State] = []
    pieces = vs.pieces.cpu().tolist()
    side = vs.side_to_move.cpu().tolist()
    cr = vs.castling.cpu().tolist()
    ep = vs.en_passant.cpu().tolist()
    hmc = vs.halfmove_clock.cpu().tolist()
    fmn = vs.fullmove_number.cpu().tolist()
    for i in range(vs.batch_size):
        out.append(
            State(
                board=list(pieces[i]),
                side_to_move=side[i],
                castling=cr[i],
                en_passant=ep[i],
                halfmove_clock=hmc[i],
                fullmove_number=fmn[i],
            )
        )
    return out


def starting_state(batch_size: int, device: torch.device | str = "cpu") -> VState:
    return from_states([State.starting() for _ in range(batch_size)], device=device)


# ---------------------------------------------------------------------------
# Plane derivation: [B, 64] piece codes -> [B, 12, 64] bool planes.
# ---------------------------------------------------------------------------


def to_planes(pieces: Tensor) -> Tensor:
    """[B, 64] int -> [B, 12, 64] bool (one-hot over piece-codes 1..12)."""
    B, _ = pieces.shape
    planes = torch.zeros(B, NUM_PLANES, 64, dtype=torch.bool, device=pieces.device)
    # piece codes 1..12 map to plane 0..11
    p = pieces.long()
    valid = p > 0
    plane_idx = (p - 1).clamp_min(0)
    # scatter: planes[b, plane_idx[b, sq], sq] = valid[b, sq]
    batch_idx = torch.arange(B, device=pieces.device).unsqueeze(1).expand(-1, 64)
    sq_idx = torch.arange(64, device=pieces.device).unsqueeze(0).expand(B, -1)
    planes[batch_idx[valid], plane_idx[valid], sq_idx[valid]] = True
    return planes


# ---------------------------------------------------------------------------
# Attack-map computation. attacked_by[B, 64] = squares attacked by `by_color`.
# ---------------------------------------------------------------------------


def _slider_attacks(
    pieces88: Tensor,  # [B, 8, 8] bool: slider locations (e.g. R+Q for orthogonal)
    empty88: Tensor,  # [B, 8, 8] bool: empty squares
    dirs: tuple[tuple[int, int], ...],
) -> Tensor:
    """Standard ray-fill: for each direction, walk up to 7 squares; each
    intermediate square that is empty extends the ray; the first non-empty
    square is "attacked" but blocks further extension.
    """
    attacks = torch.zeros_like(pieces88)
    for df, dr in dirs:
        ray = _shift(pieces88, df, dr)
        attacks = attacks | ray
        ray = ray & empty88
        for _ in range(6):
            ray = _shift(ray, df, dr)
            attacks = attacks | ray
            ray = ray & empty88
    return attacks


def attacked_squares(planes: Tensor, by_color: int) -> Tensor:
    """Compute [B, 64] bool: squares attacked by any piece of `by_color`,
    given the full 12-plane occupancy."""
    B = planes.shape[0]
    occupancy = planes.any(dim=1)  # [B, 64]
    occupancy88 = occupancy.view(B, 8, 8)
    empty88 = ~occupancy88

    if by_color == WHITE:
        pawn_t = _WPAWN_ATTACK
        N, B_, R_, Q_, K_, P_ = (
            planes[:, 1],
            planes[:, 2],
            planes[:, 3],
            planes[:, 4],
            planes[:, 5],
            planes[:, 0],
        )
    else:
        pawn_t = _BPAWN_ATTACK
        N, B_, R_, Q_, K_, P_ = (
            planes[:, 7],
            planes[:, 8],
            planes[:, 9],
            planes[:, 10],
            planes[:, 11],
            planes[:, 6],
        )

    # Knight, king, pawn: row-vector @ from-attack table.
    pawn_t_dev = pawn_t.to(planes.device)
    knight_t_dev = _KNIGHT_TABLE.to(planes.device)
    king_t_dev = _KING_TABLE.to(planes.device)

    # P/N/K planes are [B, 64] bool. Convert via float matmul -> bool.
    p_attacks = (P_.float() @ pawn_t_dev.float()) > 0  # [B, 64]
    n_attacks = (N.float() @ knight_t_dev.float()) > 0
    k_attacks = (K_.float() @ king_t_dev.float()) > 0

    # Sliders. Combine R+Q for orthogonal, B+Q for diagonal.
    rq88 = (R_ | Q_).view(B, 8, 8)
    bq88 = (B_ | Q_).view(B, 8, 8)
    orth = _slider_attacks(rq88, empty88, _QUEEN_SHIFTS[0:8:2])  # N, E, S, W (indices 0,2,4,6)
    diag = _slider_attacks(bq88, empty88, _QUEEN_SHIFTS[1:8:2])  # NE, SE, SW, NW

    return p_attacks | n_attacks | k_attacks | orth.view(B, 64) | diag.view(B, 64)


# ---------------------------------------------------------------------------
# Action-plane geometry tables.
# Build, for each (from_sq, plane), the implied (to_sq, dr_in_mover_frame,
# is_promotion, promotion_piece, plane_kind). Constants are precomputed once.
# ---------------------------------------------------------------------------


def _build_plane_geometry() -> dict[str, Tensor]:
    """Return tensors describing each (from_sq, plane) action slot:
    to_sq:        [64, 73] long, -1 if off-board.
    df, dr:       [64, 73] long, signed deltas in mover's POV (white frame).
    kind:         [64, 73] long, 0=queen-like, 1=knight, 2=underpromo.
    ray_dir:      [64, 73] long, 0..7 for queen-like planes else -1.
    ray_dist:     [64, 73] long, 1..7 for queen-like planes else -1.
    promo:        [64, 73] long, 0 or 2..4 (N,B,R) for underpromo planes.
    """
    to_sq = torch.full((64, 73), -1, dtype=torch.long)
    df = torch.zeros(64, 73, dtype=torch.long)
    dr = torch.zeros(64, 73, dtype=torch.long)
    kind = torch.zeros(64, 73, dtype=torch.long)
    ray_dir = torch.full((64, 73), -1, dtype=torch.long)
    ray_dist = torch.full((64, 73), -1, dtype=torch.long)
    promo = torch.zeros(64, 73, dtype=torch.long)

    for from_sq in range(64):
        f0, r0 = from_sq & 7, from_sq >> 3
        for d_idx, (qdf, qdr) in enumerate(_QUEEN_SHIFTS):
            for dist in range(1, 8):
                plane = d_idx * 7 + (dist - 1)
                f1 = f0 + qdf * dist
                r1 = r0 + qdr * dist
                df[from_sq, plane] = qdf * dist
                dr[from_sq, plane] = qdr * dist
                kind[from_sq, plane] = 0
                ray_dir[from_sq, plane] = d_idx
                ray_dist[from_sq, plane] = dist
                if 0 <= f1 < 8 and 0 <= r1 < 8:
                    to_sq[from_sq, plane] = r1 * 8 + f1
        for k_idx, (kdf, kdr) in enumerate(_KNIGHT_SHIFTS):
            plane = 56 + k_idx
            f1, r1 = f0 + kdf, r0 + kdr
            df[from_sq, plane] = kdf
            dr[from_sq, plane] = kdr
            kind[from_sq, plane] = 1
            if 0 <= f1 < 8 and 0 <= r1 < 8:
                to_sq[from_sq, plane] = r1 * 8 + f1
        for piece_idx, ppiece in enumerate((2, 3, 4)):  # N, B, R
            for file_idx, file_d in enumerate((-1, 0, 1)):
                plane = 64 + piece_idx * 3 + file_idx
                f1, r1 = f0 + file_d, r0 + 1
                df[from_sq, plane] = file_d
                dr[from_sq, plane] = 1
                kind[from_sq, plane] = 2
                promo[from_sq, plane] = ppiece
                if 0 <= f1 < 8 and 0 <= r1 < 8:
                    to_sq[from_sq, plane] = r1 * 8 + f1

    return {
        "to_sq": to_sq,
        "df": df,
        "dr": dr,
        "kind": kind,
        "ray_dir": ray_dir,
        "ray_dist": ray_dist,
        "promo": promo,
    }


_PLANE_GEOM = _build_plane_geometry()


def _plane_geom_on(device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device) for k, v in _PLANE_GEOM.items()}


# ---------------------------------------------------------------------------
# Pseudo-legal mask generation.
# ---------------------------------------------------------------------------


def _gather_to_pieces(pieces: Tensor, to_sq: Tensor) -> Tensor:
    """pieces: [B, 64] long; to_sq: [64, 73] long with -1 off-board.
    Returns [B, 64, 73] long: piece at to_sq, or 0 where to_sq == -1.
    """
    B = pieces.shape[0]
    safe = to_sq.clamp(min=0)  # [64, 73]
    flat = safe.reshape(-1)  # [64*73]
    gathered = pieces[:, flat]  # [B, 64*73]
    gathered = gathered.view(B, 64, 73)
    gathered = torch.where(
        to_sq.unsqueeze(0).expand(B, -1, -1) >= 0,
        gathered,
        torch.zeros_like(gathered),
    )
    return gathered


def pseudo_legal_mask(vs: VState) -> Tensor:
    """Return [B, 64, 73] bool pseudo-legal mask (does not filter for own
    king left in check). Castling moves have all rules baked in (rook
    presence, empty squares, intermediate squares not attacked, king not in
    check)."""
    device = vs.device
    B = vs.batch_size
    pieces = vs.pieces.long()  # [B, 64]
    planes = to_planes(vs.pieces)  # [B, 12, 64]

    geom = _plane_geom_on(device)
    to_sq = geom["to_sq"]  # [64, 73]
    df = geom["df"]
    dr = geom["dr"]
    kind = geom["kind"]
    ray_dir = geom["ray_dir"]
    ray_dist = geom["ray_dist"]

    # mover_white[B, 1] / mover_black[B, 1]
    side = vs.side_to_move.long()  # [B]
    mover_white = (side == WHITE).view(B, 1, 1)
    mover_black = ~mover_white

    # piece-at-from broadcast to [B, 64, 1]
    from_piece = pieces.view(B, 64, 1)

    # color of moving piece; -1 means empty.
    from_is_white = (from_piece >= 1) & (from_piece <= 6)
    from_is_black = (from_piece >= 7) & (from_piece <= 12)

    # Only squares whose piece matches side-to-move can be a move-from.
    from_belongs_to_mover = (from_is_white & mover_white) | (from_is_black & mover_black)

    # Piece type at from (1..6, P/N/B/R/Q/K), 0 if empty.
    pt = torch.where(
        from_is_black,
        from_piece - 6,
        torch.where(from_is_white, from_piece, torch.zeros_like(from_piece)),
    )  # [B, 64, 1]

    # Mover-frame (df, dr): for black, rank delta is mirrored.
    plane_df = df.unsqueeze(0).expand(B, -1, -1)  # [B, 64, 73]
    plane_dr_signed = torch.where(
        mover_white,
        dr.unsqueeze(0).expand(B, -1, -1),
        -dr.unsqueeze(0).expand(B, -1, -1),
    )
    plane_to_white = (
        torch.arange(64, device=device).view(1, 64, 1) + plane_df + 8 * dr.unsqueeze(0)
    )  # for white. Actually we need the to_sq in the "real" board frame.
    # to_sq table is defined in white POV; convert for black by reflecting rank.
    base_to = to_sq.unsqueeze(0).expand(B, -1, -1)  # [B, 64, 73]
    # Build a mirrored to_sq for black: same file, mirror rank. Squares with -1 stay -1.
    mirror_rank_to = torch.where(
        base_to >= 0,
        ((7 - (base_to >> 3)) << 3) | (base_to & 7),
        base_to,
    )
    # Also mirror from in the same way for black -- but we don't store from
    # by lookup, just apply mirroring symmetrically. Effectively we want:
    #   real_to_sq = (white frame to_sq table evaluated from from_sq), but
    # from_sq itself in white frame for white, mirrored for black.
    # Simpler: rebuild to_sq[B, 64, 73] by computing from each *real* from
    # square with the *mover-frame* dr.
    real_from = torch.arange(64, device=device).view(1, 64, 1)  # [1, 64, 1]
    real_f0 = real_from & 7
    real_r0 = real_from >> 3
    real_f1 = real_f0 + plane_df
    real_r1 = real_r0 + plane_dr_signed
    on_board = (real_f1 >= 0) & (real_f1 < 8) & (real_r1 >= 0) & (real_r1 < 8)
    real_to = real_r1 * 8 + real_f1
    real_to = torch.where(on_board, real_to, torch.full_like(real_to, -1))
    # (free unused intermediates so static analyzers don't complain)
    del base_to, mirror_rank_to, plane_to_white

    # Piece at the to-square (0 where off-board).
    safe_to = real_to.clamp(min=0)
    flat = safe_to.view(B, -1)  # [B, 64*73]
    gather = pieces.gather(1, flat).view(B, 64, 73)
    to_piece = torch.where(on_board, gather, torch.zeros_like(gather))

    to_is_white = (to_piece >= 1) & (to_piece <= 6)
    to_is_black = (to_piece >= 7) & (to_piece <= 12)
    to_is_empty = to_piece == 0
    to_is_enemy = (to_is_white & mover_black) | (to_is_black & mover_white)

    # Ray-blocker test (used for queen-like planes only). For each ray
    # direction, accumulate "blocker exists in [1, dist-1]" across distances.
    occupancy = pieces > 0  # [B, 64]
    occupancy88 = occupancy.view(B, 8, 8)

    # Compute, for each (from_sq, dir, dist), whether *any intermediate*
    # square along the ray is occupied. We do this by scanning out from
    # `from_is_mover` (a single piece per from-sq isn't necessary; we
    # actually want the test: from each from-square, does the ray contain
    # a blocker before `dist` squares?).
    #
    # Trick: for each direction, build a [B, 8, 8] tensor of "occupied
    # squares pulled back along the ray", at distances 1..6. Then the
    # blocker-flag for (from, dist) = OR over k in [1, dist-1] of
    # (occupied at from + k*step).
    #
    # We package the result as `blocker_le_dist[B, 64, dir, dist]` where
    # blocker_le_dist[..., dist] = OR_{k=1..dist} occupied(from + k*step).
    # Then "blocker before dist" = blocker_le_dist[..., dist-1] and
    # "captured-square is to_sq itself" handled via to_is_empty/enemy.

    blockers = torch.zeros(B, 64, 8, 8, dtype=torch.bool, device=device)  # [B, 64, dir, k=1..7]
    # We'll fill index k=0..6 for distances 1..7. blockers[..., d, k] =
    # is square at distance (k+1) along direction d occupied?
    for d_idx, (qdf, qdr) in enumerate(_QUEEN_SHIFTS):
        # Mirror dr for black movers: but we've kept dr in mover-frame already
        # for the action geometry. Here we work in board-frame; we need to
        # build per-batch per-direction blocker arrays for *both* frames.
        # Simpler: detect blockers in the board frame for each direction,
        # and apply mover-frame mirroring lookup later.
        cur = occupancy88
        # shifted[k] = occupied square at offset (k+1) along (qdf, qdr) from
        # the perspective of the from-square. Since out[r, f] = in[r-dr, f-df]
        # for shift(df, dr), shifting once gives "occupied at (r-dr, f-df)"
        # at position (r, f). We want "from sq (r, f), is square at
        # (r + (k+1)*qdr, f + (k+1)*qdf) occupied?". Pulling that value back
        # to position (r, f) requires shifting by (-(k+1)*qdf, -(k+1)*qdr)...
        # actually shift(df, dr) returns out[r, f] = in[r-dr, f-df], so to
        # get value at (r + step, f + step) we shift by (-step_df, -step_dr).
        # We'll iterate: cur stores the value k+1 squares away along (qdf, qdr).
        cur = _shift(occupancy88, -qdf, -qdr)
        for k in range(7):
            blockers[:, :, d_idx, k] = cur.reshape(B, 64)
            cur = _shift(cur, -qdf, -qdr)

    # Now build "any blocker strictly between from and to" at distance `dist`,
    # for each (from, plane). Need to mirror direction lookup for black.
    # Direction in board-frame for black is the direction with dr negated.
    # _QUEEN_SHIFTS pairs: (df, dr). The "negated-dr" entry is at idx mapping
    # (df, dr) -> (df, -dr): N<->S, NE<->SE, E<->E, NW<->SW.
    DIR_FLIP_RANK = torch.tensor([4, 3, 2, 1, 0, 7, 6, 5], device=device)  # 0..7 -> mirror

    plane_dir = ray_dir.unsqueeze(0).expand(B, -1, -1)  # [B, 64, 73] (plane-frame dir)
    plane_dist = ray_dist.unsqueeze(0).expand(B, -1, -1)  # [B, 64, 73]
    # Convert plane-dir (mover frame) to board-frame direction:
    board_dir = torch.where(
        mover_white.expand_as(plane_dir),
        plane_dir,
        DIR_FLIP_RANK[plane_dir.clamp_min(0)],
    )

    # Look up "occupancy at distance k from from-sq in direction d":
    # Need blockers[B, from, d, k]. We'll gather along d and k.
    # Build "any-blocker strictly between" via cumulative OR over k=0..dist-2.
    cum_blocker = blockers.clone()
    for k in range(1, 7):
        cum_blocker[:, :, :, k] = cum_blocker[:, :, :, k - 1] | cum_blocker[:, :, :, k]
    # cum_blocker[..., k] = "occupied at any distance in [1..k+1]".
    # We want "occupied at any distance in [1..dist-1]"; index = dist-2 (clamp).
    blocker_idx = (plane_dist - 2).clamp_min(0)  # [B, 64, 73]
    # Gather over dim "d" then over dim "k" -- since plane_dir varies per
    # (from, plane), use advanced indexing.
    # We need cum_blocker[b, from, board_dir, blocker_idx].
    bf_idx = torch.arange(B, device=device).view(B, 1, 1)
    sq_idx = torch.arange(64, device=device).view(1, 64, 1)
    safe_dir = board_dir.clamp_min(0)
    safe_blk_idx = blocker_idx.clamp_min(0)
    blocker_lookup = cum_blocker[bf_idx, sq_idx, safe_dir, safe_blk_idx]  # [B, 64, 73]
    has_intermediate = (plane_dist >= 2) & blocker_lookup
    is_queen_plane = kind.unsqueeze(0) == 0

    # ---- Per-piece-type plane validity. ----
    # Whether each (df, dr, kind) plane index is even *attempted* by a piece
    # of a given type. Plane geometry is from_sq-independent (deltas only
    # depend on the plane), so we read row 0 as the canonical descriptor.
    PIECE_PLANE = torch.zeros(7, 73, dtype=torch.bool, device=device)
    p_kind = kind[0]

    # Knight (2): kind == 1.
    PIECE_PLANE[2] = p_kind == 1
    # King (6): queen-like with dist == 1.
    king_planes = (p_kind == 0) & (ray_dist[0] == 1)
    PIECE_PLANE[6] = king_planes
    # Bishop (3): queen-like, diagonal directions (1, 3, 5, 7).
    diag_dirs = torch.tensor([1, 3, 5, 7], device=device)
    is_diag = (p_kind == 0) & torch.isin(ray_dir[0], diag_dirs)
    PIECE_PLANE[3] = is_diag
    # Rook (4): queen-like, orthogonal (0, 2, 4, 6).
    orth_dirs = torch.tensor([0, 2, 4, 6], device=device)
    is_orth = (p_kind == 0) & torch.isin(ray_dir[0], orth_dirs)
    PIECE_PLANE[4] = is_orth
    # Queen (5): all queen-like.
    PIECE_PLANE[5] = p_kind == 0
    # Pawn (1): more complex - dr in mover-frame must be 1 or 2 (push) or 1
    # (capture). dist-2 push only from rank 2. Promotions (queen) are queen-
    # like dist-1 pushes from rank 7. Underpromotions are kind==2.
    # We enable broadly here; finer constraints (e.g. starting rank) are
    # applied later.
    pawn_basic = (
        ((p_kind == 0) & (ray_dist[0] == 1) & (ray_dir[0] == 0))  # forward 1 (mover-frame N)
        | ((p_kind == 0) & (ray_dist[0] == 2) & (ray_dir[0] == 0))  # forward 2
        | (
            (p_kind == 0) & (ray_dist[0] == 1) & ((ray_dir[0] == 1) | (ray_dir[0] == 7))
        )  # capture diag
        | (p_kind == 2)
    )
    PIECE_PLANE[1] = pawn_basic
    # Castling: planes for E/W dist-2 from king's square. This is a king move.
    # Treated below as a special case.

    # piece-allowed mask: [B, 64, 73]
    piece_idx = pt.clamp_min(0).clamp_max(6)  # 0..6 (0 = empty)
    plane_allowed = PIECE_PLANE[piece_idx.squeeze(-1)]  # [B, 64, 73]

    # Compose pseudo-legal mask:
    # 1. From square belongs to mover.
    # 2. Plane is allowed for that piece type.
    # 3. To-square is on the board.
    # 4. To-square is empty or enemy (not own piece).
    # 5. For sliders: no intermediate blocker (and ray must extend to to_sq).
    # 6. Pawn-specific: capture vs. push semantics.
    # 7. King: castling planes are special-cased below.

    base = (
        from_belongs_to_mover.expand_as(plane_allowed)
        & plane_allowed
        & on_board
        & (to_is_empty | to_is_enemy)
    )

    # Slider intermediate-blocker filter (queen-like planes with dist >= 2).
    base = base & ~(is_queen_plane.expand_as(has_intermediate) & has_intermediate)

    # Pawn-specific filtering.
    # Identify pawn planes:
    pawn_push1 = (p_kind == 0) & (ray_dir[0] == 0) & (ray_dist[0] == 1)  # [73]
    pawn_push2 = (p_kind == 0) & (ray_dir[0] == 0) & (ray_dist[0] == 2)
    pawn_cap_e = (
        (p_kind == 0) & (ray_dir[0] == 1) & (ray_dist[0] == 1)
    )  # NE in mover-frame (right capture)
    pawn_cap_w = (
        (p_kind == 0) & (ray_dir[0] == 7) & (ray_dist[0] == 1)
    )  # NW in mover-frame (left capture)
    pawn_underpromo = p_kind == 2  # [73]

    is_pawn = pt == 1  # [B, 64, 1]
    is_pawn_b = is_pawn.expand(B, 64, 73)

    # Underpromotion can be either push (file_d=0) or capture (file_d != 0).
    underpromo_push = pawn_underpromo & (df[0] == 0)
    underpromo_cap = pawn_underpromo & (df[0] != 0)
    push_planes = pawn_push1 | pawn_push2 | underpromo_push  # [73]
    cap_planes = pawn_cap_e | pawn_cap_w | underpromo_cap  # [73]
    push_planes_b = push_planes.view(1, 1, 73)
    cap_planes_b = cap_planes.view(1, 1, 73)

    # Forbid pawn pushes onto non-empty squares.
    bad_push = is_pawn_b & push_planes_b & ~to_is_empty
    base = base & ~bad_push

    # Forbid pawn captures onto own/empty (allow enemy or en-passant target).
    ep_target = vs.en_passant.long().view(B, 1, 1).expand(B, 64, 73)
    is_ep_target = (real_to == ep_target) & (ep_target >= 0)
    bad_cap = is_pawn_b & cap_planes_b & ~(to_is_enemy | is_ep_target)
    base = base & ~bad_cap

    # Forbid double pushes from non-starting rank, and require square-1 empty.
    from_rank_mover_frame = torch.where(
        mover_white.expand(B, 64, 1),
        sq_idx >> 3,
        7 - (sq_idx >> 3),
    )  # [B, 64, 1]
    bad_push2_rank = is_pawn_b & pawn_push2.view(1, 1, 73) & (from_rank_mover_frame != 1)
    base = base & ~bad_push2_rank
    # Square between from and to must be empty.
    # Intermediate square = from + 1 step forward in mover-frame.
    forward = torch.where(
        mover_white.expand(B, 64, 1).squeeze(-1),
        torch.full((B, 64), 8, device=device, dtype=torch.long),
        torch.full((B, 64), -8, device=device, dtype=torch.long),
    )  # [B, 64]
    inter_sq = (sq_idx.squeeze(-1) + forward).clamp(0, 63)  # [B, 64]
    inter_piece = pieces.gather(1, inter_sq)  # [B, 64]
    inter_empty = (inter_piece == 0).view(B, 64, 1)
    bad_push2_blocked = is_pawn_b & pawn_push2.view(1, 1, 73) & ~inter_empty
    base = base & ~bad_push2_blocked

    # Promotions: for the queen-promote plane (push1 or NE/NW cap with dist 1)
    # the to-rank must be the last rank (rank 7 mover-frame). For
    # underpromotion we already encoded dr=1; require dest rank to be 7 in
    # mover-frame.
    to_rank_mover_frame = torch.where(
        mover_white.expand_as(real_to),
        real_to >> 3,
        7 - (real_to >> 3),
    )
    # Underpromotion plane requires to_rank_mf == 7.
    bad_underpromo = (
        is_pawn_b & pawn_underpromo.view(1, 1, 73) & (to_rank_mover_frame != 7) & on_board
    )
    base = base & ~bad_underpromo
    # Note: a non-promotion pawn can use the queen-like push-1/cap plane to
    # *queen-promote* implicitly (we accept it). We don't enforce that "push1
    # to last rank must be a queen-promo" -- it's still a legal action.

    # ---- King-specific: castling. ----
    # Castling moves are encoded as the king moving 2 squares E or W.
    # In our planes: from king's square, queen-like dir=2 (E) or 6 (W), dist=2.
    castle_E = (p_kind == 0) & (ray_dir[0] == 2) & (ray_dist[0] == 2)  # [73]
    castle_W = (p_kind == 0) & (ray_dir[0] == 6) & (ray_dist[0] == 2)
    is_king = pt == 6
    is_king_b = is_king.expand(B, 64, 73)

    # The base-mask check above might allow these planes only if the
    # 2-square ray is clear, which is partially right. We need to enforce
    # the *full* castling rules: king on starting square, rook present,
    # right not lost, intermediate squares empty (already in base mask),
    # king not in check, king does not pass through attacked square, king
    # does not land on attacked square.
    # We disable the "naive" king dist-2 moves and re-enable specifically
    # the legal castles.
    is_castle_plane = (castle_E | castle_W).view(1, 1, 73)
    naive_castle = is_king_b & is_castle_plane
    base = base & ~naive_castle

    # Compute castling legality. For each batch:
    # White kingside: king on e1 (sq 4), rook on h1 (sq 7), CR_WK set,
    # squares f1, g1 empty, e1/f1/g1 not attacked by black, side==white.
    # Mirrors for the other three.
    opp_attacks_white_to_move = attacked_squares(planes, by_color=BLACK)  # [B, 64]
    opp_attacks_black_to_move = attacked_squares(planes, by_color=WHITE)  # [B, 64]
    opp_attacks = torch.where(
        (side == WHITE).view(B, 1),
        opp_attacks_white_to_move,
        opp_attacks_black_to_move,
    )  # [B, 64]

    e1, f1, g1, d1, c1, b1, a1, h1 = 4, 5, 6, 3, 2, 1, 0, 7
    e8, f8, g8, d8, c8, b8, a8, h8 = 60, 61, 62, 59, 58, 57, 56, 63

    cr = vs.castling.long()
    side_white = side == WHITE
    side_black = ~side_white

    def _has_right(bit: int) -> Tensor:
        return ((cr >> bit) & 1).bool()

    castle_wk_ok = (
        side_white
        & _has_right(CR_WK)
        & (pieces[:, e1] == Piece.WK.value)
        & (pieces[:, h1] == Piece.WR.value)
        & (pieces[:, f1] == 0)
        & (pieces[:, g1] == 0)
        & ~opp_attacks[:, e1]
        & ~opp_attacks[:, f1]
        & ~opp_attacks[:, g1]
    )
    castle_wq_ok = (
        side_white
        & _has_right(CR_WQ)
        & (pieces[:, e1] == Piece.WK.value)
        & (pieces[:, a1] == Piece.WR.value)
        & (pieces[:, d1] == 0)
        & (pieces[:, c1] == 0)
        & (pieces[:, b1] == 0)
        & ~opp_attacks[:, e1]
        & ~opp_attacks[:, d1]
        & ~opp_attacks[:, c1]
    )
    castle_bk_ok = (
        side_black
        & _has_right(CR_BK)
        & (pieces[:, e8] == Piece.BK.value)
        & (pieces[:, h8] == Piece.BR.value)
        & (pieces[:, f8] == 0)
        & (pieces[:, g8] == 0)
        & ~opp_attacks[:, e8]
        & ~opp_attacks[:, f8]
        & ~opp_attacks[:, g8]
    )
    castle_bq_ok = (
        side_black
        & _has_right(CR_BQ)
        & (pieces[:, e8] == Piece.BK.value)
        & (pieces[:, a8] == Piece.BR.value)
        & (pieces[:, d8] == 0)
        & (pieces[:, c8] == 0)
        & (pieces[:, b8] == 0)
        & ~opp_attacks[:, e8]
        & ~opp_attacks[:, d8]
        & ~opp_attacks[:, c8]
    )

    # Place these flags into the (from_sq, plane) action slots.
    # E-castle for white: from=e1, plane=castle_E_plane_idx.
    plane_castle_E = (
        (torch.tensor(_QUEEN_SHIFTS, device=device)[:, 0] == 1)
        & (torch.tensor(_QUEEN_SHIFTS, device=device)[:, 1] == 0)
    ).nonzero()[0].item() * 7 + 1
    plane_castle_W = (
        (torch.tensor(_QUEEN_SHIFTS, device=device)[:, 0] == -1)
        & (torch.tensor(_QUEEN_SHIFTS, device=device)[:, 1] == 0)
    ).nonzero()[0].item() * 7 + 1

    castle_mask = torch.zeros(B, 64, 73, dtype=torch.bool, device=device)
    castle_mask[:, e1, plane_castle_E] = castle_wk_ok
    castle_mask[:, e1, plane_castle_W] = castle_wq_ok
    castle_mask[:, e8, plane_castle_E] = castle_bk_ok
    castle_mask[:, e8, plane_castle_W] = castle_bq_ok
    base = base | castle_mask

    return base


# ---------------------------------------------------------------------------
# Legal move filtering: from pseudo-legal candidates, remove those that
# leave own king in check.
# ---------------------------------------------------------------------------


def _expand_and_apply(vs: VState, pseudo: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Materialize a virtual batch of [N, ...] post-move boards from pseudo-
    legal candidates, where N = number of True entries in pseudo.

    Returns:
        env_idx: [N] long           (index into the original B envs)
        action: [N] long            (action index 0..ACTION_SIZE-1)
        post_planes: [N, 12, 64] bool   (post-move occupancy planes)
        mover_color: [N] long       (the side that moved)
        post_king_sq: [N] long      (square of mover's king post-move)
    """
    raise NotImplementedError("only used as a documentation stub")


def legal_action_mask(vs: VState) -> Tensor:
    """Return [B, ACTION_SIZE] bool mask of legal actions."""
    device = vs.device
    B = vs.batch_size
    pseudo = pseudo_legal_mask(vs)  # [B, 64, 73]

    # Find candidate (env, from, plane) triples.
    env_idx, from_sq, plane = pseudo.nonzero(as_tuple=True)
    N = env_idx.numel()
    if N == 0:
        return torch.zeros(B, ACTION_SIZE, dtype=torch.bool, device=device)

    pieces = vs.pieces.long()  # [B, 64]
    side = vs.side_to_move.long()  # [B]
    mover_color = side[env_idx]  # [N]

    geom = _plane_geom_on(device)
    # Compute real to_sq for each candidate using mover-frame dr.
    df_n = geom["df"][from_sq, plane]
    dr_n = geom["dr"][from_sq, plane]
    dr_signed = torch.where(mover_color == WHITE, dr_n, -dr_n)
    f0 = from_sq & 7
    r0 = from_sq >> 3
    f1 = f0 + df_n
    r1 = r0 + dr_signed
    to_sq = r1 * 8 + f1  # [N]

    moving_piece = pieces[env_idx, from_sq]  # [N]
    captured_piece = pieces[env_idx, to_sq]  # [N]

    pt_n = torch.where(
        moving_piece >= 7,
        moving_piece - 6,
        moving_piece,
    )  # piece-type 1..6

    # Build [N, 64] post-move pieces by copying then patching.
    post_pieces = pieces[env_idx].clone()  # [N, 64]
    arange_n = torch.arange(N, device=device)

    # En passant capture: pawn moving to ep target with empty to-sq.
    ep_target_n = vs.en_passant.long()[env_idx]
    is_pawn_n = pt_n == 1
    is_ep_capture = is_pawn_n & (to_sq == ep_target_n) & (captured_piece == 0)
    # Remove the pawn captured en passant (one rank "behind" to_sq from mover POV).
    ep_capture_sq = torch.where(
        mover_color == WHITE,
        to_sq - 8,
        to_sq + 8,
    )
    # Always a safe index (since ep_target is a rank-3 or rank-6 square).
    post_pieces[is_ep_capture.nonzero(as_tuple=True)[0], ep_capture_sq[is_ep_capture]] = 0

    # Move the piece: clear from_sq, set to_sq.
    post_pieces[arange_n, from_sq] = 0
    # Promotion handling: if pawn reaches last rank in mover frame, promote.
    to_rank_mover = torch.where(
        mover_color == WHITE,
        to_sq >> 3,
        7 - (to_sq >> 3),
    )
    is_promotion = is_pawn_n & (to_rank_mover == 7)
    plane_kind_n = geom["kind"][from_sq, plane]
    promo_n = geom["promo"][from_sq, plane]  # 0 or 2/3/4
    # Default queen for promotions on queen-like planes.
    promo_piece_type = torch.where(
        plane_kind_n == 2,
        promo_n,
        torch.full_like(promo_n, 5),  # queen
    )
    new_piece = torch.where(
        is_promotion,
        torch.where(mover_color == WHITE, promo_piece_type, promo_piece_type + 6),
        moving_piece,
    )
    post_pieces[arange_n, to_sq] = new_piece.to(post_pieces.dtype)

    # Castling: king moves 2 files horizontally. Move the rook.
    is_king_n = pt_n == 6
    df_abs = df_n.abs()
    is_castle = is_king_n & (df_abs == 2)
    # Kingside (df=+2 or -2 in mover frame)... since king move is in board frame
    # for both sides (king moves East/West regardless of color), we use the
    # board-frame df. Our action geometry stores board-frame df (since dr was
    # mirrored, but df is unchanged by color).
    # Actually wait: queen-like dirs are board-frame for white but mover-frame
    # for black? Let me re-check.
    # In encode_move/decode_move we mirror dr for black but not df. So df is
    # always board-frame. Good.
    castle_kingside = is_castle & (df_n == 2)
    castle_queenside = is_castle & (df_n == -2)
    rank_n = from_sq >> 3
    # Kingside: rook from h to f (file 7 -> 5).
    rook_from_ks = rank_n * 8 + 7
    rook_to_ks = rank_n * 8 + 5
    rook_from_qs = rank_n * 8 + 0
    rook_to_qs = rank_n * 8 + 3
    if castle_kingside.any():
        idx = castle_kingside.nonzero(as_tuple=True)[0]
        rook_piece = post_pieces[idx, rook_from_ks[idx]]
        post_pieces[idx, rook_from_ks[idx]] = 0
        post_pieces[idx, rook_to_ks[idx]] = rook_piece
    if castle_queenside.any():
        idx = castle_queenside.nonzero(as_tuple=True)[0]
        rook_piece = post_pieces[idx, rook_from_qs[idx]]
        post_pieces[idx, rook_from_qs[idx]] = 0
        post_pieces[idx, rook_to_qs[idx]] = rook_piece

    # Now, for each post-move position, check whether mover's king is attacked
    # by the *opponent*. We compute attacks-by-(1 - mover) on these N states.
    post_planes = to_planes(post_pieces)  # [N, 12, 64]

    # Find king square for the mover post-move.
    king_plane_idx = torch.where(mover_color == WHITE, 5, 11)
    # post_planes[N, 12, 64], gather plane per N.
    king_planes = post_planes.gather(1, king_plane_idx.view(N, 1, 1).expand(N, 1, 64)).squeeze(
        1
    )  # [N, 64]
    # king square = argmax along dim 1 (kings always present except in tests).
    king_sq = king_planes.long().argmax(dim=1)  # [N]

    # Attacks by opponent.
    # We need to call attacked_squares with the appropriate by_color per N.
    # Since by_color is a Python int in attacked_squares, split into two
    # passes (white-attacks and black-attacks) and combine.
    is_white_mover = mover_color == WHITE
    # Compute white-attacks for entries where mover==black (so opponent==white).
    # Compute black-attacks for entries where mover==white.
    attacks_by_white = attacked_squares(post_planes, by_color=WHITE)  # [N, 64]
    attacks_by_black = attacked_squares(post_planes, by_color=BLACK)
    opp_attacks = torch.where(
        is_white_mover.view(N, 1).expand(-1, 64),
        attacks_by_black,
        attacks_by_white,
    )  # [N, 64]
    king_in_check = opp_attacks.gather(1, king_sq.view(N, 1)).squeeze(1)  # [N]
    legal = ~king_in_check

    # Scatter results back to [B, ACTION_SIZE].
    out = torch.zeros(B, ACTION_SIZE, dtype=torch.bool, device=device)
    action_idx = from_sq * NUM_MOVE_PLANES + plane
    out[env_idx[legal], action_idx[legal]] = True
    return out


# ---------------------------------------------------------------------------
# apply_move: take an [B] long tensor of action indices and apply.
# ---------------------------------------------------------------------------


def apply_action(vs: VState, action: Tensor) -> VState:
    """Apply one action per env. `action` is [B] long."""
    device = vs.device
    B = vs.batch_size
    pieces = vs.pieces.long().clone()
    side = vs.side_to_move.long()
    cr = vs.castling.long()
    ep = vs.en_passant.long()
    hmc = vs.halfmove_clock.long()
    fmn = vs.fullmove_number.long()

    geom = _plane_geom_on(device)

    from_sq = action // NUM_MOVE_PLANES
    plane = action % NUM_MOVE_PLANES
    df_n = geom["df"][from_sq, plane]
    dr_n = geom["dr"][from_sq, plane]
    dr_signed = torch.where(side == WHITE, dr_n, -dr_n)
    f0 = from_sq & 7
    r0 = from_sq >> 3
    f1 = f0 + df_n
    r1 = r0 + dr_signed
    to_sq = r1 * 8 + f1

    arange_b = torch.arange(B, device=device)
    moving_piece = pieces[arange_b, from_sq]
    captured_piece = pieces[arange_b, to_sq]
    pt = torch.where(moving_piece >= 7, moving_piece - 6, moving_piece)

    is_pawn = pt == 1
    is_king = pt == 6

    # En passant capture.
    is_ep_capture = is_pawn & (to_sq == ep) & (captured_piece == 0) & (ep >= 0)
    ep_capture_sq = torch.where(side == WHITE, to_sq - 8, to_sq + 8)
    if is_ep_capture.any():
        idx = is_ep_capture.nonzero(as_tuple=True)[0]
        # Track the en-passant capture as a "capture" for the halfmove clock.
        captured_piece = captured_piece.clone()
        captured_piece[idx] = pieces[idx, ep_capture_sq[idx]]
        pieces[idx, ep_capture_sq[idx]] = 0

    # Move piece.
    pieces[arange_b, from_sq] = 0

    # Promotion: queen by default for queen-like planes; underpromo otherwise.
    plane_kind = geom["kind"][from_sq, plane]
    promo = geom["promo"][from_sq, plane]
    to_rank_mover = torch.where(side == WHITE, to_sq >> 3, 7 - (to_sq >> 3))
    is_promotion = is_pawn & (to_rank_mover == 7)
    promo_pt = torch.where(plane_kind == 2, promo, torch.full_like(promo, 5))
    new_piece = torch.where(
        is_promotion,
        torch.where(side == WHITE, promo_pt, promo_pt + 6),
        moving_piece,
    )
    pieces[arange_b, to_sq] = new_piece.to(pieces.dtype)

    # Castling: relocate rook.
    is_castle = is_king & (df_n.abs() == 2)
    castle_ks = is_castle & (df_n == 2)
    castle_qs = is_castle & (df_n == -2)
    rank_b = from_sq >> 3
    if castle_ks.any():
        idx = castle_ks.nonzero(as_tuple=True)[0]
        rk = rank_b[idx]
        rook = pieces[idx, rk * 8 + 7]
        pieces[idx, rk * 8 + 7] = 0
        pieces[idx, rk * 8 + 5] = rook
    if castle_qs.any():
        idx = castle_qs.nonzero(as_tuple=True)[0]
        rk = rank_b[idx]
        rook = pieces[idx, rk * 8 + 0]
        pieces[idx, rk * 8 + 0] = 0
        pieces[idx, rk * 8 + 3] = rook

    # Update castling rights.
    # King move clears both rights for that side.
    new_cr = cr.clone()
    moved_king = is_king
    # White king moved.
    wk_moved = moved_king & (side == WHITE)
    bk_moved = moved_king & (side == BLACK)
    new_cr = torch.where(wk_moved, new_cr & ~((1 << CR_WK) | (1 << CR_WQ)), new_cr)
    new_cr = torch.where(bk_moved, new_cr & ~((1 << CR_BK) | (1 << CR_BQ)), new_cr)
    # Rook leaving its starting square or being captured there clears that bit.
    # Squares (a1, h1, a8, h8) <-> (CR_WQ, CR_WK, CR_BQ, CR_BK).
    rook_sq_to_bit = {0: CR_WQ, 7: CR_WK, 56: CR_BQ, 63: CR_BK}
    for sq_val, bit in rook_sq_to_bit.items():
        affected = (from_sq == sq_val) | (to_sq == sq_val)
        new_cr = torch.where(affected, new_cr & ~(1 << bit), new_cr)

    # Update en passant.
    is_double_push = is_pawn & (df_n == 0) & (dr_n == 2)
    new_ep = torch.where(
        is_double_push,
        torch.where(side == WHITE, to_sq - 8, to_sq + 8),
        torch.full_like(ep, -1),
    )

    # Update halfmove clock.
    is_capture = (captured_piece != 0) | is_ep_capture
    new_hmc = torch.where(is_pawn | is_capture, torch.zeros_like(hmc), hmc + 1)

    # Update fullmove number (increments after black's move).
    new_fmn = torch.where(side == BLACK, fmn + 1, fmn)

    new_side = (1 - side).to(torch.int8)

    return VState(
        pieces=pieces.to(torch.int8),
        side_to_move=new_side,
        castling=new_cr.to(torch.int8),
        en_passant=new_ep.to(torch.int8),
        halfmove_clock=new_hmc.to(torch.int16),
        fullmove_number=new_fmn.to(torch.int16),
    )


# ---------------------------------------------------------------------------
# Insufficient-material and game result.
# ---------------------------------------------------------------------------


def _insufficient_material(pieces: Tensor) -> Tensor:
    """[B, 64] long -> [B] bool."""
    device = pieces.device
    # Pawns / rooks / queens of either color => sufficient.
    sufficient = (
        (pieces == Piece.WP.value)
        | (pieces == Piece.BP.value)
        | (pieces == Piece.WR.value)
        | (pieces == Piece.BR.value)
        | (pieces == Piece.WQ.value)
        | (pieces == Piece.BQ.value)
    ).any(dim=1)
    # Count knights and bishops per color.
    wn = (pieces == Piece.WN.value).sum(dim=1)
    bn = (pieces == Piece.BN.value).sum(dim=1)
    wb = (pieces == Piece.WB.value).sum(dim=1)
    bb = (pieces == Piece.BB.value).sum(dim=1)

    minor_count = wn + bn + wb + bb

    # K vs K
    only_kings = minor_count == 0
    # K + 1 minor vs K
    one_minor = minor_count == 1
    # K + B vs K + B with same-color bishops
    bishops_equal_one = (wb == 1) & (bb == 1) & (wn == 0) & (bn == 0)
    # Determine bishop square colors.
    sq_colors = torch.tensor(
        [((sq >> 3) + (sq & 7)) & 1 for sq in range(64)],
        device=device,
        dtype=torch.long,
    )
    wb_sq = pieces == Piece.WB.value
    bb_sq = pieces == Piece.BB.value
    # Color of (only) white bishop, of (only) black bishop. Use argmax (only
    # valid if exactly one bishop of that color present).
    wb_color = (wb_sq.float() * sq_colors.float()).sum(dim=1)
    bb_color = (bb_sq.float() * sq_colors.float()).sum(dim=1)
    bishops_same_color = bishops_equal_one & (wb_color == bb_color)

    insuff = ~sufficient & (only_kings | one_minor | bishops_same_color)
    return insuff


def game_result(vs: VState) -> Tensor:
    """[B] long: ONGOING/WHITE_WIN/BLACK_WIN/DRAW."""
    device = vs.device
    B = vs.batch_size
    legal = legal_action_mask(vs)  # [B, ACTION_SIZE]
    has_move = legal.any(dim=1)  # [B]

    # Is mover currently in check?
    pieces = vs.pieces.long()
    planes = to_planes(vs.pieces)
    side = vs.side_to_move.long()
    king_plane_idx = torch.where(side == WHITE, 5, 11)
    king_planes = planes.gather(1, king_plane_idx.view(B, 1, 1).expand(B, 1, 64)).squeeze(1)
    king_sq = king_planes.long().argmax(dim=1)
    attacks_white = attacked_squares(planes, by_color=WHITE)
    attacks_black = attacked_squares(planes, by_color=BLACK)
    opp_attacks = torch.where(
        (side == WHITE).view(B, 1),
        attacks_black,
        attacks_white,
    )
    in_check = opp_attacks.gather(1, king_sq.view(B, 1)).squeeze(1)

    # 50-move rule and insufficient material.
    fifty = vs.halfmove_clock.long() >= 100
    insuff = _insufficient_material(pieces)

    result = torch.full((B,), ONGOING, dtype=torch.long, device=device)
    # No legal move:
    no_move_check = ~has_move & in_check
    no_move_stale = ~has_move & ~in_check
    result = torch.where(
        no_move_check,
        torch.where(
            side == WHITE,
            torch.tensor(BLACK_WIN, device=device),
            torch.tensor(WHITE_WIN, device=device),
        ),
        result,
    )
    result = torch.where(no_move_stale, torch.tensor(DRAW, device=device), result)
    # 50-move / insufficient material draws (only if game still ongoing AND
    # we have a legal move -- otherwise checkmate takes priority).
    draw_other = has_move & (fifty | insuff)
    result = torch.where(draw_other, torch.tensor(DRAW, device=device), result)
    return result
