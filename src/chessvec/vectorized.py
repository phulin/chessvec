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
    """[B, 64] int -> [B, 12, 64] bool (one-hot over piece-codes 1..12).

    Branchless via a broadcast equality (no data-dependent indexing) so the
    function is safe under torch.compile / CUDA Graph capture.
    """
    B = pieces.shape[0]
    codes = torch.arange(1, NUM_PLANES + 1, device=pieces.device, dtype=pieces.dtype).view(
        1, NUM_PLANES, 1
    )
    return pieces.view(B, 1, 64) == codes  # [B, 12, 64] bool


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
    tables = _jump_tables_on(planes.device)
    pawn_t_dev = tables["wpawn"] if by_color == WHITE else tables["bpawn"]
    knight_t_dev = tables["knight"]
    king_t_dev = tables["king"]
    del pawn_t  # no longer used; keep the if-branch above for symmetry

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


def attacked_squares_both(planes: Tensor) -> Tensor:
    """Return [B, 2, 64] bool: attacks_by_white at index 0, attacks_by_black at 1.

    Stacks white/black slider inputs along the batch dim so that the slider
    ray-fill loop runs once on a 2B-sized tensor rather than twice on B-sized
    tensors. Halves the kernel-launch count of `attacked_squares` when both
    colors are needed.
    """
    B = planes.shape[0]
    occupancy = planes.any(dim=1)  # [B, 64]
    occupancy88 = occupancy.view(B, 8, 8)
    empty88 = ~occupancy88

    tables = _jump_tables_on(planes.device)
    knight_t = tables["knight"].float()
    king_t = tables["king"].float()
    wpawn_t = tables["wpawn"].float()
    bpawn_t = tables["bpawn"].float()

    # White pieces.
    P_w, N_w, B_w, R_w, Q_w, K_w = (
        planes[:, 0], planes[:, 1], planes[:, 2], planes[:, 3], planes[:, 4], planes[:, 5],
    )
    # Black pieces.
    P_b, N_b, B_b, R_b, Q_b, K_b = (
        planes[:, 6], planes[:, 7], planes[:, 8], planes[:, 9], planes[:, 10], planes[:, 11],
    )

    p_w = (P_w.float() @ wpawn_t) > 0
    n_w = (N_w.float() @ knight_t) > 0
    k_w = (K_w.float() @ king_t) > 0
    p_b = (P_b.float() @ bpawn_t) > 0
    n_b = (N_b.float() @ knight_t) > 0
    k_b = (K_b.float() @ king_t) > 0

    # Stack slider sources along batch dim and run one scan each for orth/diag.
    rq_w = (R_w | Q_w).view(B, 8, 8)
    bq_w = (B_w | Q_w).view(B, 8, 8)
    rq_b = (R_b | Q_b).view(B, 8, 8)
    bq_b = (B_b | Q_b).view(B, 8, 8)
    rq_stack = torch.cat([rq_w, rq_b], dim=0)  # [2B, 8, 8]
    bq_stack = torch.cat([bq_w, bq_b], dim=0)
    empty_stack = empty88.repeat(2, 1, 1)  # [2B, 8, 8]
    orth = _slider_attacks(rq_stack, empty_stack, _QUEEN_SHIFTS[0:8:2])
    diag = _slider_attacks(bq_stack, empty_stack, _QUEEN_SHIFTS[1:8:2])
    orth_w, orth_b = orth[:B], orth[B:]
    diag_w, diag_b = diag[:B], diag[B:]

    attacks_w = p_w | n_w | k_w | orth_w.view(B, 64) | diag_w.view(B, 64)
    attacks_b = p_b | n_b | k_b | orth_b.view(B, 64) | diag_b.view(B, 64)
    return torch.stack([attacks_w, attacks_b], dim=1)  # [B, 2, 64]


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

# Castle plane indices in the 73-plane layout. Queen-like planes are laid out
# as dir_idx * 7 + (dist - 1). E = dir 2, W = dir 6, dist = 2.
_PLANE_CASTLE_E = 2 * 7 + 1  # 15
_PLANE_CASTLE_W = 6 * 7 + 1  # 43


_PLANE_GEOM_CACHE: dict[torch.device, dict[str, Tensor]] = {}


def _plane_geom_on(device: torch.device) -> dict[str, Tensor]:
    cached = _PLANE_GEOM_CACHE.get(device)
    if cached is None:
        cached = {k: v.to(device) for k, v in _PLANE_GEOM.items()}
        _PLANE_GEOM_CACHE[device] = cached
    return cached


# Cache device-resident copies of the precomputed jump/attack tables that
# attacked_squares moves to the device on every call.
_JUMP_TABLES_CACHE: dict[torch.device, dict[str, Tensor]] = {}


def _jump_tables_on(device: torch.device) -> dict[str, Tensor]:
    cached = _JUMP_TABLES_CACHE.get(device)
    if cached is None:
        cached = {
            "king": _KING_TABLE.to(device),
            "knight": _KNIGHT_TABLE.to(device),
            "wpawn": _WPAWN_ATTACK.to(device),
            "bpawn": _BPAWN_ATTACK.to(device),
            # Source-of-attack table: source[from] gives squares (true at sq) from
            # which a piece-of-this-type at sq attacks `from`. For knight/king the
            # tables are symmetric; for pawns we transpose.
            "wpawn_sources": _WPAWN_ATTACK.t().contiguous().to(device),
            "bpawn_sources": _BPAWN_ATTACK.t().contiguous().to(device),
        }
        _JUMP_TABLES_CACHE[device] = cached
    return cached


def _build_ray_from_sq() -> Tensor:
    """[64, 8, 7] long: square at step (k+1) along direction d from `from_sq`,
    or -1 if off-board."""
    tbl = torch.full((64, 8, 7), -1, dtype=torch.long)
    for from_sq in range(64):
        f0, r0 = from_sq & 7, from_sq >> 3
        for d, (df, dr) in enumerate(_QUEEN_SHIFTS):
            for k in range(7):
                f = f0 + df * (k + 1)
                r = r0 + dr * (k + 1)
                if 0 <= f < 8 and 0 <= r < 8:
                    tbl[from_sq, d, k] = r * 8 + f
    return tbl


_RAY_FROM_SQ = _build_ray_from_sq()
_RAY_FROM_SQ_CACHE: dict[torch.device, Tensor] = {}


def _ray_from_sq_on(device: torch.device) -> Tensor:
    cached = _RAY_FROM_SQ_CACHE.get(device)
    if cached is None:
        cached = _RAY_FROM_SQ.to(device)
        _RAY_FROM_SQ_CACHE[device] = cached
    return cached


# Constants used by the pin/check legality computation, also cached per device.
_PIN_CONST_CACHE: dict[torch.device, dict[str, Tensor]] = {}


def _pin_constants_on(device: torch.device) -> dict[str, Tensor]:
    cached = _PIN_CONST_CACHE.get(device)
    if cached is None:
        # is_orth_dir: True for N/E/S/W (indices 0,2,4,6), False for diagonals.
        is_orth = torch.tensor(
            [True, False, True, False, True, False, True, False], device=device
        )
        # Per-side piece codes indexed by side (0=white, 1=black). Used to
        # avoid per-call `torch.full_like(side, code)` allocations.
        piece_codes = {
            "king": torch.tensor([Piece.WK.value, Piece.BK.value], device=device, dtype=torch.long),
            "knight_enemy": torch.tensor(
                [Piece.BN.value, Piece.WN.value], device=device, dtype=torch.long
            ),
            "pawn_enemy": torch.tensor(
                [Piece.BP.value, Piece.WP.value], device=device, dtype=torch.long
            ),
            "pawn_mover": torch.tensor(
                [Piece.WP.value, Piece.BP.value], device=device, dtype=torch.long
            ),
        }
        cached = {"is_orth_dir": is_orth, **piece_codes}
        _PIN_CONST_CACHE[device] = cached
    return cached


# ---------------------------------------------------------------------------
# Pseudo-legal mask generation.
# ---------------------------------------------------------------------------


def _slider_blockers(pieces: Tensor, B: int, device: torch.device) -> Tensor:
    """Compute [B, 64, 8, 7] cumulative slider-ray blockers. Triton fast path
    on CUDA, PyTorch shift-loop fallback elsewhere."""
    if device.type == "cuda":
        try:
            from .triton_step import _HAS_TRITON, triton_slider_blockers
        except ImportError:
            _HAS_TRITON = False
        if _HAS_TRITON:
            return triton_slider_blockers(pieces)
    # PyTorch fallback: same shift-based scan as before, output cumulative.
    occupancy88 = (pieces > 0).view(B, 8, 8)
    raw = torch.zeros(B, 64, 8, 7, dtype=torch.bool, device=device)
    for d_idx, (qdf, qdr) in enumerate(_QUEEN_SHIFTS):
        cur = _shift(occupancy88, -qdf, -qdr)
        for k in range(7):
            raw[:, :, d_idx, k] = cur.reshape(B, 64)
            cur = _shift(cur, -qdf, -qdr)
    cum = raw.clone()
    for k in range(1, 7):
        cum[:, :, :, k] = cum[:, :, :, k - 1] | cum[:, :, :, k]
    return cum


def _enemy_attacks_xray(
    vs: VState, pieces: Tensor, side: Tensor, king_sq: Tensor, B: int, device: torch.device
) -> Tensor:
    """Compute [B, 64] enemy_attacks_xray (king removed). Uses the Triton
    kernel on CUDA when available; falls back to PyTorch elsewhere.
    """
    if device.type == "cuda":
        try:
            from .triton_step import _HAS_TRITON, triton_enemy_attacks_xray
        except ImportError:
            _HAS_TRITON = False
        if _HAS_TRITON:
            return triton_enemy_attacks_xray(vs)
    pieces_no_king = pieces.clone()
    pieces_no_king[torch.arange(B, device=device), king_sq] = 0
    planes_no_king = to_planes(pieces_no_king.to(torch.int8))
    attacks_xray_both = attacked_squares_both(planes_no_king)  # [B, 2, 64]
    enemy_color_idx = (1 - side).view(B, 1, 1).expand(B, 1, 64)
    return attacks_xray_both.gather(1, enemy_color_idx).squeeze(1)


def _move_rook_branchless(
    pieces: Tensor, mask: Tensor, src_sq: Tensor, dst_sq: Tensor
) -> None:
    """Where mask[i] is true, move pieces[i, src_sq[i]] to pieces[i, dst_sq[i]],
    zeroing the source. No-op (but still touches both squares) elsewhere.
    src_sq / dst_sq must be valid indices for every row regardless of mask
    (callers pass rank-derived squares that are always in [0, 63]).
    """
    src = src_sq.view(-1, 1)
    dst = dst_sq.view(-1, 1)
    src_val = pieces.gather(1, src).squeeze(1)
    cur_dst = pieces.gather(1, dst).squeeze(1)
    pieces.scatter_(1, src, torch.where(mask, torch.zeros_like(src_val), src_val).view(-1, 1))
    pieces.scatter_(1, dst, torch.where(mask, src_val, cur_dst).view(-1, 1))


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


def pseudo_legal_mask(
    vs: VState,
    _opp_attacks: Tensor | None = None,
) -> Tensor:
    """Return [B, 64, 73] bool pseudo-legal mask (does not filter for own
    king left in check). Castling moves have all rules baked in (rook
    presence, empty squares, intermediate squares not attacked, king not in
    check).

    `_opp_attacks` is an optional precomputed [B, 64] bool map of squares
    attacked by the opponent (the side NOT to move). When supplied (e.g. by
    `legal_action_mask`, which already needs this map for king-in-check
    detection), it is used directly for the castling-through-check test
    instead of being recomputed.
    """
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

    cum_blocker = _slider_blockers(vs.pieces, B, device)  # [B, 64, 8, 7] cumulative

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

    # cum_blocker[B, 64, 8, 7] already cumulative-OR'd by `_slider_blockers`.
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
    if _opp_attacks is None:
        attacks_both = attacked_squares_both(planes)  # [B, 2, 64]
        enemy_color_idx = (1 - side).view(B, 1, 1).expand(B, 1, 64)
        opp_attacks = attacks_both.gather(1, enemy_color_idx).squeeze(1)
    else:
        opp_attacks = _opp_attacks

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
    castle_mask = torch.zeros(B, 64, 73, dtype=torch.bool, device=device)
    castle_mask[:, e1, _PLANE_CASTLE_E] = castle_wk_ok
    castle_mask[:, e1, _PLANE_CASTLE_W] = castle_wq_ok
    castle_mask[:, e8, _PLANE_CASTLE_E] = castle_bk_ok
    castle_mask[:, e8, _PLANE_CASTLE_W] = castle_bq_ok
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
    """Return [B, ACTION_SIZE] bool mask of legal actions.

    Uses pin/check-based legality (no make-test): determine checkers, pins,
    and a king-safety attack map once per board, then filter pseudo-legal
    moves against those constraints. Avoids the O(B*K*64) post-move expansion
    of the make-test approach.
    """
    device = vs.device
    if device.type == "cuda":
        try:
            from .triton_step import _HAS_TRITON, triton_legal_action_mask
        except ImportError:
            _HAS_TRITON = False
        if _HAS_TRITON:
            return triton_legal_action_mask(vs)
    B = vs.batch_size
    pieces = vs.pieces.long()  # [B, 64]
    side = vs.side_to_move.long()  # [B]
    is_white_mover_b = (side == WHITE).view(B, 1)
    is_white_mover_3 = is_white_mover_b.view(B, 1, 1)

    # ---- King square (direct equality, no full to_planes materialization) ----
    pin_const = _pin_constants_on(device)
    king_code = pin_const["king"][side]  # [B]
    king_planes = pieces == king_code.view(B, 1)  # [B, 64] bool
    king_sq = king_planes.long().argmax(dim=1)  # [B]

    # ---- Enemy attack map with mover's king removed (xray) ----
    # Computed up-front so we can share it with pseudo_legal_mask's castling
    # check (which is also more correct: castling f1/g1 must not be attacked
    # *after* the king vacates e1 -- i.e., with the king removed).
    enemy_attacks_xray = _enemy_attacks_xray(vs, pieces, side, king_sq, B, device)

    # ---- Pseudo-legal candidates ----
    pseudo = pseudo_legal_mask(vs, _opp_attacks=enemy_attacks_xray)  # [B, 64, 73]

    # ---- Recompute real_to[B, 64, 73] (in board frame) ----
    geom = _plane_geom_on(device)
    df_g = geom["df"]  # [64, 73]
    dr_g = geom["dr"]  # [64, 73]
    plane_df = df_g.unsqueeze(0).expand(B, -1, -1)
    plane_dr = torch.where(is_white_mover_3, dr_g.unsqueeze(0), -dr_g.unsqueeze(0))
    sq_arr = torch.arange(64, device=device).view(1, 64, 1)
    real_f0 = sq_arr & 7
    real_r0 = sq_arr >> 3
    real_f1 = real_f0 + plane_df
    real_r1 = real_r0 + plane_dr
    on_board = (real_f1 >= 0) & (real_f1 < 8) & (real_r1 >= 0) & (real_r1 < 8)
    real_to = torch.where(on_board, real_r1 * 8 + real_f1, torch.full_like(real_f1, 0))
    # [B, 64, 73] long; sentinel 0 for off-board (gated by `on_board`).

    # ---- Ray analysis from king ----
    ray_table = _ray_from_sq_on(device)  # [64, 8, 7]
    ray_squares = ray_table[king_sq]  # [B, 8, 7] long, -1 if off-board
    safe_ray = ray_squares.clamp(min=0)
    ray_pieces = pieces.gather(1, safe_ray.view(B, -1)).view(B, 8, 7)
    ray_pieces = torch.where(ray_squares >= 0, ray_pieces, torch.zeros_like(ray_pieces))
    ray_occ = ray_pieces > 0
    cum_occ = ray_occ.long().cumsum(dim=2)
    first_piece_mask = (cum_occ == 1) & ray_occ  # [B, 8, 7]
    second_piece_mask = (cum_occ == 2) & ray_occ

    ray_is_white = (ray_pieces >= 1) & (ray_pieces <= 6)
    ray_is_black = (ray_pieces >= 7) & (ray_pieces <= 12)
    ray_is_own = torch.where(is_white_mover_3, ray_is_white, ray_is_black)
    ray_is_orth_slider = (
        (ray_pieces == Piece.WR.value)
        | (ray_pieces == Piece.WQ.value)
        | (ray_pieces == Piece.BR.value)
        | (ray_pieces == Piece.BQ.value)
    )
    ray_is_diag_slider = (
        (ray_pieces == Piece.WB.value)
        | (ray_pieces == Piece.WQ.value)
        | (ray_pieces == Piece.BB.value)
        | (ray_pieces == Piece.BQ.value)
    )
    is_orth_dir = pin_const["is_orth_dir"].view(1, 8, 1)  # [1, 8, 1]
    ray_slider_d = torch.where(is_orth_dir, ray_is_orth_slider, ray_is_diag_slider)
    ray_is_enemy = torch.where(is_white_mover_3, ray_is_black, ray_is_white)
    ray_enemy_slider_d = ray_slider_d & ray_is_enemy  # [B, 8, 7]

    first_is_enemy_slider = (first_piece_mask & ray_enemy_slider_d).any(dim=2)  # [B, 8]
    first_is_own = (first_piece_mask & ray_is_own).any(dim=2)
    second_is_enemy_slider = (second_piece_mask & ray_enemy_slider_d).any(dim=2)
    pin_in_dir = first_is_own & second_is_enemy_slider  # [B, 8]

    # Step indices and squares for first/second pieces along each direction.
    step_first = first_piece_mask.long().argmax(dim=2)  # [B, 8] in 0..6
    step_second = second_piece_mask.long().argmax(dim=2)  # [B, 8]
    first_piece_sq = ray_squares.gather(2, step_first.unsqueeze(-1)).squeeze(-1)  # [B, 8]
    step_idx = torch.arange(7, device=device).view(1, 1, 7)  # [1, 1, 7]

    # ---- Checkers ----
    tables = _jump_tables_on(device)
    knight_attack_to_king = tables["knight"][king_sq]  # [B, 64]
    enemy_knight_plane = pieces == pin_const["knight_enemy"][side].view(B, 1)
    knight_checkers = knight_attack_to_king & enemy_knight_plane

    pawn_attack_to_king = torch.where(
        is_white_mover_b,
        tables["bpawn_sources"][king_sq],  # for white mover, enemy is black
        tables["wpawn_sources"][king_sq],
    )
    enemy_pawn_plane = pieces == pin_const["pawn_enemy"][side].view(B, 1)
    pawn_checkers = pawn_attack_to_king & enemy_pawn_plane

    # Slider checkers: scatter first_piece_sq into a [B, 64] mask where
    # first_is_enemy_slider holds.
    slider_buf = torch.zeros(B, 65, dtype=torch.bool, device=device)
    slider_check_src = torch.where(
        first_is_enemy_slider,
        first_piece_sq,
        torch.full_like(first_piece_sq, 64),
    )
    slider_buf.scatter_(
        1, slider_check_src, torch.ones_like(slider_check_src, dtype=torch.bool)
    )
    slider_checkers = slider_buf[:, :64]

    checkers_mask = knight_checkers | pawn_checkers | slider_checkers
    num_checkers = checkers_mask.long().sum(dim=1)  # [B]

    # ---- Block-or-capture mask (only meaningful if num_checkers == 1) ----
    in_segment = step_idx <= step_first.view(B, 8, 1)  # [B, 8, 7]
    block_mask = (
        in_segment & first_is_enemy_slider.view(B, 8, 1) & (ray_squares >= 0)
    )
    block_sq_safe = torch.where(block_mask, ray_squares, torch.full_like(ray_squares, 64))
    block_buf = torch.zeros(B, 65, dtype=torch.bool, device=device)
    block_buf.scatter_(
        1, block_sq_safe.view(B, -1), torch.ones((B, 8 * 7), dtype=torch.bool, device=device)
    )
    block_or_capture = block_buf[:, :64] | knight_checkers | pawn_checkers  # [B, 64]

    # ---- Pin destinations per pinned-from square (fully vectorized) ----
    # pin_legal[b, from_sq, to_sq]: True iff move from→to allowed by pin.
    # Build allowed-destination bitmap per direction in one shot, then write
    # all 8 direction-rows into pin_legal via a single scatter (no Python loop).
    in_pin_segment = step_idx <= step_second.view(B, 8, 1)  # [B, 8, 7]
    pin_valid = in_pin_segment & pin_in_dir.view(B, 8, 1) & (ray_squares >= 0)  # [B, 8, 7]
    pin_sq = torch.where(pin_valid, ray_squares, torch.full_like(ray_squares, 64))  # [B, 8, 7]
    allowed_buf = torch.zeros(B, 8, 65, dtype=torch.bool, device=device)
    allowed_buf.scatter_(2, pin_sq, torch.ones_like(pin_sq, dtype=torch.bool))
    allowed_per_d = allowed_buf[:, :, :64]  # [B, 8, 64]

    # Route writes for non-pinned (b, d) to row 64 (sentinel) so they don't
    # disturb real from-square rows. A piece can be pinned in at most one
    # direction, so per-row writes have no real conflicts.
    pinned_row = torch.where(
        pin_in_dir, first_piece_sq, torch.full_like(first_piece_sq, 64)
    )  # [B, 8]
    pin_legal_buf = torch.ones(B, 65, 64, dtype=torch.bool, device=device)
    pin_legal_buf.scatter_(
        1, pinned_row.view(B, 8, 1).expand(B, 8, 64), allowed_per_d
    )
    pin_legal = pin_legal_buf[:, :64, :]

    # ---- Per-candidate legality (broadcast over [B, 64, 73]) ----
    from_piece_b = pieces.view(B, 64, 1)  # [B, 64, 1]
    from_is_king = (from_piece_b == Piece.WK.value) | (from_piece_b == Piece.BK.value)

    safe_to = real_to  # already clamped to 0; gated by on_board

    # King moves: to_sq must not be attacked by enemy in the king-removed map.
    enemy_xray_at_to = enemy_attacks_xray.gather(1, safe_to.view(B, -1)).view(B, 64, 73)
    king_move_legal = ~enemy_xray_at_to

    # Non-king moves:
    # - Forbidden in double check.
    # - Must address single check (block or capture the checker).
    # - Must respect pin ray for pinned pieces.
    not_double_check = (num_checkers <= 1).view(B, 1, 1)
    in_check_b = (num_checkers >= 1).view(B, 1, 1)
    block_at_to = block_or_capture.gather(1, safe_to.view(B, -1)).view(B, 64, 73)
    check_resolve_ok = ~in_check_b | block_at_to  # if in check, must resolve
    # pin_legal[b, from_sq, to_sq]
    pin_legal_at_to = pin_legal.gather(2, safe_to)  # [B, 64, 73]
    non_king_legal = not_double_check & pin_legal_at_to & check_resolve_ok

    # Castling moves are king moves whose to_sq attack-checks were already
    # baked into pseudo_legal_mask. Keep them: castle from-sq is the king,
    # so they go through the king_move_legal path (which uses xray attacks).
    legal = torch.where(from_is_king.expand_as(non_king_legal), king_move_legal, non_king_legal)

    # ---- En-passant horizontal pin: removing two adjacent pawns on the
    # capture rank can expose the king to an enemy rook/queen on that rank.
    # Detect: mover pawn moves to ep_target. After removing both pawns from
    # `from_sq` and `ep_capture_sq` (= ep_target ∓ 8), is king attacked by
    # an enemy R/Q on the same rank as ep_capture_sq?
    ep_target = vs.en_passant.long()  # [B], -1 if none
    ep_cap_sq = torch.where(side == WHITE, ep_target - 8, ep_target + 8).clamp(0, 63)
    mover_pawn = pieces == pin_const["pawn_mover"][side].view(B, 1)  # [B, 64]
    is_ep_move = (
        pseudo
        & (real_to == ep_target.view(B, 1, 1))
        & (ep_target.view(B, 1, 1) >= 0)
        & on_board
        & mover_pawn.view(B, 64, 1)
    )
    ep_safe = _ep_horizontal_safe(pieces, king_sq, ep_cap_sq, side, is_ep_move)
    legal = legal & (~is_ep_move | ep_safe)

    return (pseudo & legal).view(B, ACTION_SIZE)


def _ep_horizontal_safe(
    pieces: Tensor,
    king_sq: Tensor,
    ep_cap_sq: Tensor,
    side: Tensor,
    is_ep_move: Tensor,
) -> Tensor:
    """Per (b, from_sq, plane): is the en-passant capture safe from a horizontal
    pin? Returns [B, 64, 73] bool (True for non-ep moves and safe ep moves).
    """
    device = pieces.device
    B = pieces.shape[0]
    # For each ep move, the "vacated" squares are from_sq and ep_cap_sq.
    # Walk west and east from king on its rank. The first non-vacated piece
    # in each direction must NOT be enemy rook/queen (else king is exposed).
    king_rank = king_sq >> 3  # [B]
    king_file = king_sq & 7  # [B]
    enemy_rq = torch.where(
        (side == WHITE).view(B, 1),
        ((pieces == Piece.BR.value) | (pieces == Piece.BQ.value)),
        ((pieces == Piece.WR.value) | (pieces == Piece.WQ.value)),
    )  # [B, 64]

    # Build per (b, from_sq) "vacated" mask: from_sq + ep_cap_sq.
    arange64 = torch.arange(64, device=device).view(1, 64, 1)
    sq_idx_full = torch.arange(64, device=device).view(1, 1, 64)  # [1, 1, 64]
    # vacated[b, from_sq, sq] = (sq == from_sq) | (sq == ep_cap_sq[b])
    vacated_from = sq_idx_full == arange64  # [1, 64, 64]
    vacated_epcap = sq_idx_full == ep_cap_sq.view(B, 1, 1)  # [B, 1, 64]
    vacated = vacated_from | vacated_epcap  # [B, 64, 64]

    # Squares on king's rank: rank == king_rank[b].
    rank_of_sq = sq_idx_full >> 3  # [1, 1, 64]
    on_king_rank = rank_of_sq == king_rank.view(B, 1, 1)  # [B, 1, 64]
    file_of_sq = sq_idx_full & 7
    file_diff = file_of_sq - king_file.view(B, 1, 1)  # [B, 1, 64]

    # Occupancy on king's rank, excluding vacated squares (and excluding king).
    occ = (pieces > 0).view(B, 1, 64) & on_king_rank & ~vacated
    occ = occ & (sq_idx_full != king_sq.view(B, 1, 1))

    # West side (file_diff < 0): find first occupied square with greatest file.
    west_occ = occ & (file_diff < 0)
    east_occ = occ & (file_diff > 0)
    # First occupied west: max file among west_occ squares.
    # Use file_of_sq with sentinel where not occupied.
    west_file = torch.where(west_occ, file_of_sq, torch.full_like(file_of_sq, -1))
    west_first_file = west_file.max(dim=2).values  # [B, 64]
    # Square = king_rank * 8 + west_first_file
    has_west = west_first_file >= 0  # [B, 64]
    west_sq = (king_rank.view(B, 1) * 8 + west_first_file.clamp(min=0))  # [B, 64]
    west_is_enemy_rq = enemy_rq.gather(1, west_sq) & has_west  # [B, 64]

    east_file = torch.where(east_occ, file_of_sq, torch.full_like(file_of_sq, 8))
    east_first_file = east_file.min(dim=2).values  # [B, 64]
    has_east = east_first_file < 8
    east_sq = (king_rank.view(B, 1) * 8 + east_first_file.clamp(max=7))
    east_is_enemy_rq = enemy_rq.gather(1, east_sq) & has_east

    exposed = west_is_enemy_rq | east_is_enemy_rq  # [B, 64]
    # ep is safe iff NOT exposed (per-from_sq).
    ep_safe = ~exposed  # [B, 64]
    return ep_safe.view(B, 64, 1).expand(B, 64, 73) | ~is_ep_move


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

    # En passant capture. Update branchlessly to avoid a CPU sync on .any().
    is_ep_capture = is_pawn & (to_sq == ep) & (captured_piece == 0) & (ep >= 0)
    ep_capture_sq = torch.where(side == WHITE, to_sq - 8, to_sq + 8).clamp(0, 63)
    ep_pawn = pieces.gather(1, ep_capture_sq.view(B, 1)).squeeze(1)
    captured_piece = torch.where(is_ep_capture, ep_pawn, captured_piece)
    pieces.scatter_(
        1,
        ep_capture_sq.view(B, 1),
        torch.where(is_ep_capture, torch.zeros_like(ep_pawn), ep_pawn).view(B, 1),
    )

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
    _move_rook_branchless(pieces, castle_ks, rank_b * 8 + 7, rank_b * 8 + 5)
    _move_rook_branchless(pieces, castle_qs, rank_b * 8 + 0, rank_b * 8 + 3)

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
# Fused step + torch.compile helper.
# ---------------------------------------------------------------------------


# Register VState with PyTorch's pytree machinery so torch.compile can flatten
# it across function boundaries. Using the _pytree API directly because VState
# is a plain dataclass (not a NamedTuple) and we want stable field order.
def _vstate_flatten(vs: "VState"):
    return (
        [
            vs.pieces,
            vs.side_to_move,
            vs.castling,
            vs.en_passant,
            vs.halfmove_clock,
            vs.fullmove_number,
        ],
        None,
    )


def _vstate_unflatten(values, _ctx):
    return VState(*values)


try:
    import torch.utils._pytree as _pytree

    _pytree.register_pytree_node(VState, _vstate_flatten, _vstate_unflatten)
except (ImportError, ValueError):
    # Already registered, or pytree API not available — both are fine.
    pass


def step(vs: VState, action: Tensor) -> tuple[VState, Tensor]:
    """One environment step: apply `action` per env, then return the new
    state together with its legal-action mask. Side-effect-free.
    """
    new_vs = apply_action(vs, action)
    legal = legal_action_mask(new_vs)
    return new_vs, legal


def make_step_compiled(
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
):
    """Return a torch.compile'd version of :func:`step`.

    On accelerators with `mode="reduce-overhead"` PyTorch will additionally
    capture CUDA Graphs once shapes stabilize, eliminating per-iteration
    kernel-launch overhead. `fullgraph=True` opts into stricter tracing
    (good for catching graph breaks but can fail on dynamic ops).

    The returned callable wraps the compiled function with a
    `cudagraph_mark_step_begin()` so that each invocation gets a fresh
    output buffer from the captured graph (otherwise outputs from the
    previous call would be silently overwritten when re-fed as inputs).
    """
    compiled = torch.compile(step, mode=mode, fullgraph=fullgraph, dynamic=False)

    def _wrapped(vs: VState, action: Tensor) -> tuple[VState, Tensor]:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        out_vs, out_mask = compiled(vs, action)
        # Clone outputs so the next call's inputs don't alias the captured
        # graph's output buffers (which would be overwritten on the next run).
        cloned = VState(
            pieces=out_vs.pieces.clone(),
            side_to_move=out_vs.side_to_move.clone(),
            castling=out_vs.castling.clone(),
            en_passant=out_vs.en_passant.clone(),
            halfmove_clock=out_vs.halfmove_clock.clone(),
            fullmove_number=out_vs.fullmove_number.clone(),
        )
        return cloned, out_mask.clone()

    return _wrapped


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
    attacks_both = attacked_squares_both(planes)  # [B, 2, 64]
    enemy_color_idx = (1 - side).view(B, 1, 1).expand(B, 1, 64)
    opp_attacks = attacks_both.gather(1, enemy_color_idx).squeeze(1)
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
