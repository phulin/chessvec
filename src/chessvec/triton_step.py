"""Triton kernel that applies one action per env in a single kernel launch.

Mirrors :func:`chessvec.vectorized.apply_action`, but does the entire state
transition (board mutation, castling-rook relocation, en-passant capture,
promotion, castling-rights update, en-passant target, halfmove clock,
fullmove number, side flip) inside one Triton program per environment.

The kernel uses one program per env and a 64-wide vector for the board, so
all per-square updates are vectorized within the program.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .types import ACTION_SIZE, NUM_MOVE_PLANES
from .vectorized import VState, _PLANE_GEOM

_PADDED_PLANES = 128  # next pow2 >= 73 for the per-plane block dim

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - optional dep
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _step_body(
        pieces,    # int32 [BLOCK_SQ]
        side,      # int32 scalar
        cr,        # int32 scalar
        ep,        # int32 scalar
        hmc,       # int32 scalar
        fmn,       # int32 scalar
        action,    # int32 scalar
        df_table_ptr,
        dr_table_ptr,
        kind_table_ptr,
        promo_table_ptr,
        BLOCK_SQ: tl.constexpr,
        NUM_PLANES_C: tl.constexpr,
    ):
        """Apply one action to a single VState held in registers.

        Returns (new_pieces[BLOCK_SQ] int32, new_side, new_cr, new_ep, new_hmc, new_fmn)
        as scalars (with new_pieces a tile).
        """
        sq = tl.arange(0, BLOCK_SQ)

        from_sq = action // NUM_PLANES_C
        plane = action % NUM_PLANES_C
        tbl_idx = from_sq * NUM_PLANES_C + plane
        df = tl.load(df_table_ptr + tbl_idx).to(tl.int32)
        dr = tl.load(dr_table_ptr + tbl_idx).to(tl.int32)
        kind_v = tl.load(kind_table_ptr + tbl_idx).to(tl.int32)
        promo = tl.load(promo_table_ptr + tbl_idx).to(tl.int32)

        is_white = side == 0
        dr_signed = tl.where(is_white, dr, -dr)
        f0 = from_sq & 7
        r0 = from_sq // 8
        f1 = f0 + df
        r1 = r0 + dr_signed
        to_sq = r1 * 8 + f1

        moving_piece = tl.sum(tl.where(sq == from_sq, pieces, 0))
        captured_piece = tl.sum(tl.where(sq == to_sq, pieces, 0))
        pt = tl.where(moving_piece >= 7, moving_piece - 6, moving_piece)

        is_pawn = pt == 1
        is_king = pt == 6

        is_ep_capture = is_pawn & (to_sq == ep) & (captured_piece == 0) & (ep >= 0)
        ep_cap_sq = tl.where(is_white, to_sq - 8, to_sq + 8)

        to_rank_mover = tl.where(is_white, to_sq // 8, 7 - (to_sq // 8))
        is_promotion = is_pawn & (to_rank_mover == 7)
        promo_pt = tl.where(kind_v == 2, promo, 5)
        promoted_piece = tl.where(is_white, promo_pt, promo_pt + 6)
        new_piece = tl.where(is_promotion, promoted_piece, moving_piece)

        is_castle = is_king & ((df == 2) | (df == -2))
        castle_ks = is_castle & (df == 2)
        rank_b = from_sq // 8
        rook_from = tl.where(castle_ks, rank_b * 8 + 7, rank_b * 8 + 0)
        rook_to = tl.where(castle_ks, rank_b * 8 + 5, rank_b * 8 + 3)
        rook_piece = tl.where(is_white, 4, 10)

        new_pieces = pieces
        new_pieces = tl.where(is_ep_capture & (sq == ep_cap_sq), 0, new_pieces)
        new_pieces = tl.where(sq == from_sq, 0, new_pieces)
        new_pieces = tl.where(sq == to_sq, new_piece, new_pieces)
        new_pieces = tl.where(is_castle & (sq == rook_from), 0, new_pieces)
        new_pieces = tl.where(is_castle & (sq == rook_to), rook_piece, new_pieces)

        new_cr = cr
        wk_moved = is_king & is_white
        bk_moved = is_king & (~is_white)
        new_cr = tl.where(wk_moved, new_cr & ~0x3, new_cr)
        new_cr = tl.where(bk_moved, new_cr & ~0xC, new_cr)
        aff_a1 = (from_sq == 0) | (to_sq == 0)
        aff_h1 = (from_sq == 7) | (to_sq == 7)
        aff_a8 = (from_sq == 56) | (to_sq == 56)
        aff_h8 = (from_sq == 63) | (to_sq == 63)
        new_cr = tl.where(aff_a1, new_cr & ~0x2, new_cr)
        new_cr = tl.where(aff_h1, new_cr & ~0x1, new_cr)
        new_cr = tl.where(aff_a8, new_cr & ~0x8, new_cr)
        new_cr = tl.where(aff_h8, new_cr & ~0x4, new_cr)

        is_double_push = is_pawn & (df == 0) & (dr == 2)
        new_ep = tl.where(
            is_double_push,
            tl.where(is_white, to_sq - 8, to_sq + 8),
            -1,
        )

        is_capture = (captured_piece != 0) | is_ep_capture
        new_hmc = tl.where(is_pawn | is_capture, 0, hmc + 1)
        new_fmn = tl.where(is_white, fmn, fmn + 1)
        new_side = 1 - side

        return new_pieces, new_side, new_cr, new_ep, new_hmc, new_fmn


    @triton.jit
    def _step_kernel(
        pieces_ptr,  # int8 [B, 64]
        side_ptr,  # int8 [B]
        cr_ptr,  # int8 [B]
        ep_ptr,  # int8 [B]
        hmc_ptr,  # int16 [B]
        fmn_ptr,  # int16 [B]
        action_ptr,  # int64 [B]
        out_pieces_ptr,
        out_side_ptr,
        out_cr_ptr,
        out_ep_ptr,
        out_hmc_ptr,
        out_fmn_ptr,
        df_table_ptr,  # int64 [64*73]
        dr_table_ptr,
        kind_table_ptr,
        promo_table_ptr,
        BLOCK_SQ: tl.constexpr,  # 64
        NUM_PLANES_C: tl.constexpr,  # 73
    ):
        pid = tl.program_id(axis=0)
        sq = tl.arange(0, BLOCK_SQ)

        # ----- Load state -----
        pieces = tl.load(pieces_ptr + pid * BLOCK_SQ + sq).to(tl.int32)
        side = tl.load(side_ptr + pid).to(tl.int32)
        cr = tl.load(cr_ptr + pid).to(tl.int32)
        ep = tl.load(ep_ptr + pid).to(tl.int32)
        hmc = tl.load(hmc_ptr + pid).to(tl.int32)
        fmn = tl.load(fmn_ptr + pid).to(tl.int32)
        action = tl.load(action_ptr + pid).to(tl.int32)

        new_pieces, new_side, new_cr, new_ep, new_hmc, new_fmn = _step_body(
            pieces, side, cr, ep, hmc, fmn, action,
            df_table_ptr, dr_table_ptr, kind_table_ptr, promo_table_ptr,
            BLOCK_SQ=BLOCK_SQ, NUM_PLANES_C=NUM_PLANES_C,
        )

        tl.store(out_pieces_ptr + pid * BLOCK_SQ + sq, new_pieces.to(tl.int8))
        tl.store(out_side_ptr + pid, new_side.to(tl.int8))
        tl.store(out_cr_ptr + pid, new_cr.to(tl.int8))
        tl.store(out_ep_ptr + pid, new_ep.to(tl.int8))
        tl.store(out_hmc_ptr + pid, new_hmc.to(tl.int16))
        tl.store(out_fmn_ptr + pid, new_fmn.to(tl.int16))


if _HAS_TRITON:

    @triton.jit
    def _enemy_attacks_xray_kernel(
        pieces_ptr,  # int8 [B, 64]
        side_ptr,    # int8 [B]
        output_ptr,  # uint8 [B, 64]
        KNIGHT_TBL_ptr,  # int8 [64*64]
        KING_TBL_ptr,
        WPAWN_TBL_ptr,
        BPAWN_TBL_ptr,
        BLOCK_SQ: tl.constexpr,  # 64
    ):
        """Compute squares attacked by the side NOT to move, with the mover's
        king removed (for king-move legality / pseudo castling check).

        One program per board. Output is bool [B, 64] (uint8: 0 or 1).
        """
        pid = tl.program_id(axis=0)
        sq = tl.arange(0, BLOCK_SQ)

        pieces = tl.load(pieces_ptr + pid * BLOCK_SQ + sq).to(tl.int32)
        side = tl.load(side_ptr + pid).to(tl.int32)
        is_white_mover = side == 0

        # Remove mover's king. Piece codes: WK=6, BK=12.
        king_code = tl.where(is_white_mover, 6, 12)
        is_king_sq = pieces == king_code
        pieces_nk = tl.where(is_king_sq, 0, pieces)

        # Enemy piece codes (mover white -> enemy black starts at code 7).
        pawn_code = tl.where(is_white_mover, 7, 1)
        knight_code = tl.where(is_white_mover, 8, 2)
        bishop_code = tl.where(is_white_mover, 9, 3)
        rook_code = tl.where(is_white_mover, 10, 4)
        queen_code = tl.where(is_white_mover, 11, 5)
        enemy_king_code = tl.where(is_white_mover, 12, 6)

        enemy_pawn = (pieces_nk == pawn_code).to(tl.int32)
        enemy_knight = (pieces_nk == knight_code).to(tl.int32)
        enemy_king = (pieces_nk == enemy_king_code).to(tl.int32)

        # ---- Jump attacks (matmul over the [64,64] attack tables) ----
        rows = tl.arange(0, BLOCK_SQ)[:, None]
        cols = tl.arange(0, BLOCK_SQ)[None, :]
        KNIGHT_TBL = tl.load(KNIGHT_TBL_ptr + rows * 64 + cols).to(tl.int32)
        KING_TBL = tl.load(KING_TBL_ptr + rows * 64 + cols).to(tl.int32)
        BP_TBL = tl.load(BPAWN_TBL_ptr + rows * 64 + cols).to(tl.int32)
        WP_TBL = tl.load(WPAWN_TBL_ptr + rows * 64 + cols).to(tl.int32)
        # When mover is white, enemy is black, use BPAWN_TBL (black pawn attacks).
        PAWN_TBL = tl.where(is_white_mover, BP_TBL, WP_TBL)

        # attacks[sq] = OR_k (enemy[k] & TBL[k, sq])
        knight_atk = (tl.sum(enemy_knight[:, None] * KNIGHT_TBL, axis=0) > 0).to(tl.int32)
        king_atk = (tl.sum(enemy_king[:, None] * KING_TBL, axis=0) > 0).to(tl.int32)
        pawn_atk = (tl.sum(enemy_pawn[:, None] * PAWN_TBL, axis=0) > 0).to(tl.int32)

        # ---- Slider attacks via per-target ray walks ----
        # For each target sq, walk outward in 8 directions; first piece
        # encountered (if matching slider type for that direction) is an
        # attacker. tl.gather lets us read pieces_nk at variable indices.
        file_sq = sq & 7
        rank_sq = sq >> 3

        is_orth_slider = ((pieces_nk == rook_code) | (pieces_nk == queen_code)).to(tl.int32)
        is_diag_slider = ((pieces_nk == bishop_code) | (pieces_nk == queen_code)).to(tl.int32)

        slider_atk = tl.zeros([BLOCK_SQ], tl.int32)

        # Direction (df, dr) for each of 8 dirs; orth flag indicates rook-like.
        # Order: N, NE, E, SE, S, SW, W, NW.
        DFS = [0, 1, 1, 1, 0, -1, -1, -1]
        DRS = [1, 1, 0, -1, -1, -1, 0, 1]
        ORTH = [True, False, True, False, True, False, True, False]

        for d_idx in tl.static_range(0, 8):
            df = DFS[d_idx]
            dr = DRS[d_idx]
            is_orth = ORTH[d_idx]
            walking = tl.full([BLOCK_SQ], 1, tl.int32)
            for step in tl.static_range(1, 8):
                target_file = file_sq + df * step
                target_rank = rank_sq + dr * step
                on_board = ((target_file >= 0) & (target_file < 8) &
                            (target_rank >= 0) & (target_rank < 8)).to(tl.int32)
                target_sq = tl.where(on_board > 0, target_rank * 8 + target_file, 0)
                target_piece = tl.gather(pieces_nk, target_sq, axis=0)
                target_piece = tl.where(on_board > 0, target_piece, 0)
                is_piece = (target_piece != 0).to(tl.int32)
                first_piece = walking * is_piece
                if is_orth:
                    matches = ((target_piece == rook_code) |
                               (target_piece == queen_code)).to(tl.int32)
                else:
                    matches = ((target_piece == bishop_code) |
                               (target_piece == queen_code)).to(tl.int32)
                slider_atk = slider_atk | (first_piece * matches)
                walking = walking * (1 - is_piece) * on_board

        attacks = (knight_atk | king_atk | pawn_atk | slider_atk).to(tl.int8)
        tl.store(output_ptr + pid * BLOCK_SQ + sq, attacks)


_TABLES_CACHE: dict[torch.device, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
_ATTACK_TABLES_CACHE: dict[torch.device, tuple[Tensor, Tensor, Tensor, Tensor]] = {}


def _attack_tables_on(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if device not in _ATTACK_TABLES_CACHE:
        from .vectorized import _KING_TABLE, _KNIGHT_TABLE, _WPAWN_ATTACK, _BPAWN_ATTACK
        _ATTACK_TABLES_CACHE[device] = (
            _KNIGHT_TABLE.to(device=device, dtype=torch.int8).contiguous().view(-1),
            _KING_TABLE.to(device=device, dtype=torch.int8).contiguous().view(-1),
            _WPAWN_ATTACK.to(device=device, dtype=torch.int8).contiguous().view(-1),
            _BPAWN_ATTACK.to(device=device, dtype=torch.int8).contiguous().view(-1),
        )
    return _ATTACK_TABLES_CACHE[device]


if _HAS_TRITON:

    @triton.jit
    def _slider_blockers_kernel(
        pieces_ptr,  # int8 [B, 64]
        output_ptr,  # uint8 [B, 64, 8, 7] cumulative blockers along each ray
        BLOCK_SQ: tl.constexpr,  # 64
    ):
        """For each (board, from_sq, dir, k): is any of the squares at
        distances 1..k+1 along direction `dir` from `from_sq` occupied?
        Cumulative-OR across k. One program per board.
        """
        pid = tl.program_id(axis=0)
        sq = tl.arange(0, BLOCK_SQ)
        pieces = tl.load(pieces_ptr + pid * BLOCK_SQ + sq).to(tl.int32)

        file_sq = sq & 7
        rank_sq = sq >> 3

        DFS = [0, 1, 1, 1, 0, -1, -1, -1]
        DRS = [1, 1, 0, -1, -1, -1, 0, 1]

        base = pid * BLOCK_SQ * 8 * 7  # output offset for this board
        for d_idx in tl.static_range(0, 8):
            df = DFS[d_idx]
            dr = DRS[d_idx]
            cum = tl.zeros([BLOCK_SQ], tl.int32)
            for k in tl.static_range(0, 7):
                step = k + 1
                tf = file_sq + df * step
                tr = rank_sq + dr * step
                on_board = ((tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8)).to(tl.int32)
                target_sq = tl.where(on_board > 0, tr * 8 + tf, 0)
                target_piece = tl.gather(pieces, target_sq, axis=0)
                target_piece = tl.where(on_board > 0, target_piece, 0)
                is_blocker = ((target_piece != 0) & (on_board > 0)).to(tl.int32)
                cum = cum | is_blocker
                # Output layout: [B, 64, 8, 7] -> offset = pid*64*8*7 + sq*56 + d*7 + k.
                offset = base + sq * 56 + d_idx * 7 + k
                tl.store(output_ptr + offset, cum.to(tl.int8))


def triton_enemy_attacks_xray(vs: VState) -> Tensor:
    """Compute [B, 64] bool: squares attacked by the side NOT to move, with
    the mover's king removed. Single Triton kernel launch.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is not available")
    if vs.device.type != "cuda":
        raise RuntimeError("triton_enemy_attacks_xray requires CUDA tensors")
    B = vs.batch_size
    device = vs.device
    nt, kt, wpt, bpt = _attack_tables_on(device)
    pieces_in = vs.pieces.contiguous()
    side_in = vs.side_to_move.contiguous()
    out = torch.empty(B, 64, dtype=torch.uint8, device=device)
    _enemy_attacks_xray_kernel[(B,)](
        pieces_in, side_in, out,
        nt, kt, wpt, bpt,
        BLOCK_SQ=64,
    )
    return out.to(torch.bool)



def _tables_on(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if device not in _TABLES_CACHE:
        df = _PLANE_GEOM["df"].to(device=device, dtype=torch.int64).contiguous().view(-1)
        dr = _PLANE_GEOM["dr"].to(device=device, dtype=torch.int64).contiguous().view(-1)
        kd = _PLANE_GEOM["kind"].to(device=device, dtype=torch.int64).contiguous().view(-1)
        pr = _PLANE_GEOM["promo"].to(device=device, dtype=torch.int64).contiguous().view(-1)
        _TABLES_CACHE[device] = (df, dr, kd, pr)
    return _TABLES_CACHE[device]


def triton_step(vs: VState, action: Tensor) -> VState:
    """Apply one action per env using a single Triton kernel launch.

    Equivalent to :func:`chessvec.vectorized.apply_action`. Requires CUDA.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is not available")
    if vs.device.type != "cuda":
        raise RuntimeError("triton_step requires CUDA tensors")

    B = vs.batch_size
    device = vs.device
    df_t, dr_t, kd_t, pr_t = _tables_on(device)

    pieces_in = vs.pieces.contiguous()
    side_in = vs.side_to_move.contiguous()
    cr_in = vs.castling.contiguous()
    ep_in = vs.en_passant.contiguous()
    hmc_in = vs.halfmove_clock.contiguous()
    fmn_in = vs.fullmove_number.contiguous()
    action_in = action.to(torch.int64).contiguous()

    out_pieces = torch.empty_like(pieces_in)
    out_side = torch.empty_like(side_in)
    out_cr = torch.empty_like(cr_in)
    out_ep = torch.empty_like(ep_in)
    out_hmc = torch.empty_like(hmc_in)
    out_fmn = torch.empty_like(fmn_in)

    _step_kernel[(B,)](
        pieces_in,
        side_in,
        cr_in,
        ep_in,
        hmc_in,
        fmn_in,
        action_in,
        out_pieces,
        out_side,
        out_cr,
        out_ep,
        out_hmc,
        out_fmn,
        df_t,
        dr_t,
        kd_t,
        pr_t,
        BLOCK_SQ=64,
        NUM_PLANES_C=NUM_MOVE_PLANES,
        num_warps=2,
    )

    return VState(
        pieces=out_pieces,
        side_to_move=out_side,
        castling=out_cr,
        en_passant=out_ep,
        halfmove_clock=out_hmc,
        fullmove_number=out_fmn,
    )


# ---------------------------------------------------------------------------
# Fully-fused legal-action-mask kernel.
#
# One Triton program per board computes the entire [64, 73] legal-action mask:
#   1. find king square
#   2. enemy attacks with mover's king removed (jump pieces + ray sliders)
#   3. ray analysis from king: checkers, pin info, block-or-capture squares
#   4. per-(from_sq, plane) pseudo-legal flags (piece type + ray clearance +
#      pawn semantics + en-passant + on-board)
#   5. legality filter (king-attack/pin/check-resolve), castling, ep-h-pin
#   6. write [64, 73] -> [4672] bool out
# ---------------------------------------------------------------------------


_LEGAL_TABLES_CACHE: dict[torch.device, dict[str, Tensor]] = {}


def _legal_tables_on(device: torch.device) -> dict[str, Tensor]:
    cached = _LEGAL_TABLES_CACHE.get(device)
    if cached is not None:
        return cached

    PAD = _PADDED_PLANES  # 128
    NPL = NUM_MOVE_PLANES  # 73

    # Plane-geom tables, padded to [64, PAD]. Padded slots get sentinels that
    # cause `plane_allowed` to be False everywhere.
    def _pad(src: Tensor, fill: int) -> Tensor:
        out = torch.full((64, PAD), fill, dtype=src.dtype)
        out[:, :NPL] = src
        return out

    g = _PLANE_GEOM
    df = _pad(g["df"], 0).to(device=device, dtype=torch.int32).contiguous()
    dr = _pad(g["dr"], 0).to(device=device, dtype=torch.int32).contiguous()
    kind = _pad(g["kind"], -1).to(device=device, dtype=torch.int32).contiguous()
    ray_dir = _pad(g["ray_dir"], -1).to(device=device, dtype=torch.int32).contiguous()
    ray_dist = _pad(g["ray_dist"], -1).to(device=device, dtype=torch.int32).contiguous()
    promo = _pad(g["promo"], 0).to(device=device, dtype=torch.int32).contiguous()

    from .vectorized import _BPAWN_ATTACK, _KING_TABLE, _KNIGHT_TABLE, _WPAWN_ATTACK

    knight_tbl = _KNIGHT_TABLE.to(device=device, dtype=torch.int32).contiguous()
    king_tbl = _KING_TABLE.to(device=device, dtype=torch.int32).contiguous()
    wpawn_tbl = _WPAWN_ATTACK.to(device=device, dtype=torch.int32).contiguous()
    bpawn_tbl = _BPAWN_ATTACK.to(device=device, dtype=torch.int32).contiguous()

    # Per-source-square u64 attack bitboards (idea #1). For each [64,64] bool
    # table, pack each row into a u64 stored as signed int64 (matching torch).
    def _table_to_bb(tbl: Tensor) -> Tensor:
        out = torch.zeros(64, dtype=torch.int64)
        for s in range(64):
            v = 0
            for t in range(64):
                if bool(tbl[s, t]):
                    v |= 1 << t
            out[s] = v if v < (1 << 63) else v - (1 << 64)
        return out

    knight_bb = _table_to_bb(_KNIGHT_TABLE).to(device=device).contiguous()
    king_bb = _table_to_bb(_KING_TABLE).to(device=device).contiguous()
    wpawn_bb = _table_to_bb(_WPAWN_ATTACK).to(device=device).contiguous()
    bpawn_bb = _table_to_bb(_BPAWN_ATTACK).to(device=device).contiguous()

    # Helper: convert a Python int holding a 64-bit pattern into a signed int
    # with the same bit pattern (since torch.int64 can't directly hold values
    # >= 2**63). Used to build bitboard tables.
    def _u64_to_i64(x: int) -> int:
        return x if x < (1 << 63) else x - (1 << 64)

    # ----- Per-(from_sq, plane) "between" bitboards -----
    # bit `s` set iff square `s` is strictly between `from_sq` and the plane's
    # to-square along the plane's board-frame direction. 0 for non-queen-like
    # planes and dist <= 1 (no intermediate squares). Two tables: white uses
    # (df, dr); black uses (df, -dr).
    QUEEN_SHIFTS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    bb_data: list[list[list[int]]] = [
        [[0] * PAD for _ in range(64)] for _ in range(2)
    ]
    for is_black in (0, 1):
        for from_sq in range(64):
            f0, r0 = from_sq & 7, from_sq >> 3
            for d_idx, (qdf, qdr) in enumerate(QUEEN_SHIFTS):
                qdr_b = -qdr if is_black else qdr
                for dist in range(1, 8):
                    plane = d_idx * 7 + (dist - 1)
                    bits = 0
                    for k in range(1, dist):
                        f = f0 + qdf * k
                        r = r0 + qdr_b * k
                        if 0 <= f < 8 and 0 <= r < 8:
                            bits |= 1 << (r * 8 + f)
                    bb_data[is_black][from_sq][plane] = _u64_to_i64(bits)
    between_bb = torch.tensor(bb_data, dtype=torch.int64, device=device).contiguous()

    # ----- Plane bitmask constants & per-piece-type / per-pin lookups -----
    # Encode each as (lo, hi) int64 covering bits 0..72.
    def _bb(planes_iter):
        lo = 0
        hi = 0
        for p in planes_iter:
            if p < 64:
                lo |= 1 << p
            else:
                hi |= 1 << (p - 64)
        return (lo, hi)

    plane_q = list(range(56))         # queen-like
    plane_k = list(range(56, 64))     # knight kind
    plane_u = list(range(64, 73))     # underpromo
    plane_diag = [d * 7 + (dist - 1) for d in (1, 3, 5, 7) for dist in range(1, 8)]
    plane_orth = [d * 7 + (dist - 1) for d in (0, 2, 4, 6) for dist in range(1, 8)]

    PLANE_PUSH1 = [0 * 7 + 0]                      # forward 1
    PLANE_PUSH2 = [0 * 7 + 1]                      # forward 2
    PLANE_CAP_E = [1 * 7 + 0]                      # NE 1 (mover frame)
    PLANE_CAP_W = [7 * 7 + 0]                      # NW 1
    UNDERPROMO_PUSH = [64 + p_idx * 3 + 1 for p_idx in range(3)]  # file_d=0
    UNDERPROMO_CAP = [p for p in plane_u if p not in UNDERPROMO_PUSH]
    PUSH_PLANES = PLANE_PUSH1 + PLANE_PUSH2 + UNDERPROMO_PUSH
    CAP_PLANES = PLANE_CAP_E + PLANE_CAP_W + UNDERPROMO_CAP
    PLANE_CE_IDX = 2 * 7 + 1   # 15  (E dist 2)
    PLANE_CW_IDX = 6 * 7 + 1   # 43  (W dist 2)
    IS_CASTLE = [PLANE_CE_IDX, PLANE_CW_IDX]
    PLANE_VALID = list(range(NPL))

    # Pawn queen-like allowed: forward 1, 2 + diag captures.
    pawn_q_ok = (
        [0 * 7 + 0, 0 * 7 + 1]            # forward 1, 2
        + [1 * 7 + 0, 7 * 7 + 0]          # diag captures (mover frame)
    )
    # King queen-like dist=1 (8 directions).
    king_dist1 = [d * 7 + 0 for d in range(8)]

    # Per-piece-type allowed planes, indexed by piece type 0..6 (0 = empty).
    plane_allowed_per_pt = [
        [],                             # 0 empty
        pawn_q_ok + plane_u,            # 1 pawn
        plane_k,                        # 2 knight
        plane_diag,                     # 3 bishop
        plane_orth,                     # 4 rook
        plane_q,                        # 5 queen
        king_dist1,                     # 6 king
    ]
    plane_allowed_data = [
        [_u64_to_i64(v) for v in _bb(planes)]
        for planes in plane_allowed_per_pt
    ]
    plane_allowed_bb = torch.tensor(
        plane_allowed_data, dtype=torch.int64, device=device
    ).contiguous()

    # Per-side per-board-frame-direction plane bitmask: which planes have
    # board-frame move direction == d (or include both d and (d+4)%8 for the
    # pin-movable lookup below).
    # Direction codes:
    #   0:N(0,1) 1:NE(1,1) 2:E(1,0) 3:SE(1,-1)
    #   4:S(0,-1) 5:SW(-1,-1) 6:W(-1,0) 7:NW(-1,1)
    DIR_OF = {(0, 1): 0, (1, 1): 1, (1, 0): 2, (1, -1): 3,
              (0, -1): 4, (-1, -1): 5, (-1, 0): 6, (-1, 1): 7}

    def _board_dir_for_plane(plane: int, side_black: int) -> int:
        if plane < 56:
            d_idx, dist_m1 = divmod(plane, 7)
            df_p, dr_p = QUEEN_SHIFTS[d_idx]
            dr_b = -dr_p if side_black else dr_p
            df_u = (1 if df_p > 0 else (-1 if df_p < 0 else 0))
            dr_u = (1 if dr_b > 0 else (-1 if dr_b < 0 else 0))
            return DIR_OF.get((df_u, dr_u), -1)
        if plane < 64:
            return -1  # knight has no single direction
        # underpromo: file_d in {-1, 0, +1}, dr in mover frame = +1 → board: ±1
        local = plane - 64
        file_idx = local % 3
        df = file_idx - 1
        dr = -1 if side_black else 1
        return DIR_OF.get((df, dr), -1)

    # Pin-movable plane bitmask: indexed by [side, pin_idx, lo/hi].
    # pin_idx 0..7 = pinned along board-frame direction d. Movable planes =
    #   { p : board_dir(p, side) == d or board_dir(p, side) == (d+4) % 8 }.
    # pin_idx 8 = unpinned → all planes valid (PLANE_VALID).
    pin_movable_data = [[[0, 0] for _ in range(9)] for _ in range(2)]
    for side_black in (0, 1):
        for d in range(8):
            opp = (d + 4) % 8
            planes = [
                p for p in range(NPL)
                if _board_dir_for_plane(p, side_black) in (d, opp)
            ]
            lo, hi = _bb(planes)
            pin_movable_data[side_black][d] = [_u64_to_i64(lo), _u64_to_i64(hi)]
        lo, hi = _bb(PLANE_VALID)
        pin_movable_data[side_black][8] = [_u64_to_i64(lo), _u64_to_i64(hi)]
    pin_movable_bb = torch.tensor(
        pin_movable_data, dtype=torch.int64, device=device
    ).contiguous()

    # Per-(side, from_sq) on-board bitmask: bit `plane` set iff plane is on
    # the board from this from_sq. Replaces the per-cell on-board int math.
    # Also precompute real_to[side, from_sq, plane] (int8) — the board-frame
    # to-square for each (from_sq, plane) under the given mover side. 0 for
    # off-board (gated by on_board_bb at the use site).
    on_board_data = [[[0, 0] for _ in range(64)] for _ in range(2)]
    real_to_data = [[[0] * PAD for _ in range(64)] for _ in range(2)]
    for side_black in (0, 1):
        for from_sq in range(64):
            f0, r0 = from_sq & 7, from_sq >> 3
            planes_ok: list[int] = []
            for p in range(NPL):
                df_v = int(g["df"][from_sq, p])
                dr_v_mf = int(g["dr"][from_sq, p])
                dr_v = -dr_v_mf if side_black else dr_v_mf
                f1 = f0 + df_v
                r1 = r0 + dr_v
                if 0 <= f1 < 8 and 0 <= r1 < 8:
                    planes_ok.append(p)
                    real_to_data[side_black][from_sq][p] = r1 * 8 + f1
            lo, hi = _bb(planes_ok)
            on_board_data[side_black][from_sq] = [_u64_to_i64(lo), _u64_to_i64(hi)]
    on_board_bb = torch.tensor(
        on_board_data, dtype=torch.int64, device=device
    ).contiguous()
    real_to_tbl = torch.tensor(
        real_to_data, dtype=torch.int8, device=device
    ).contiguous()

    # Single-plane scalar bitmasks (lo, hi). Passed as constexpr ints into the
    # kernel — they're tiny and held in a single register slot.
    BB_SCALARS = {
        "is_q":   _bb(plane_q),
        "is_u":   _bb(plane_u),
        "is_castle": _bb(IS_CASTLE),
        "push": _bb(PUSH_PLANES),
        "cap":  _bb(CAP_PLANES),
        "push1": _bb(PLANE_PUSH1),
        "push2": _bb(PLANE_PUSH2),
        "underpromo_push": _bb(UNDERPROMO_PUSH),
        "plane_valid": _bb(PLANE_VALID),
    }

    cached = {
        "df": df,
        "dr": dr,
        "kind": kind,
        "ray_dir": ray_dir,
        "ray_dist": ray_dist,
        "promo": promo,
        "knight": knight_tbl,
        "king": king_tbl,
        "wpawn": wpawn_tbl,
        "bpawn": bpawn_tbl,
        "knight_bb": knight_bb,
        "king_bb": king_bb,
        "wpawn_bb": wpawn_bb,
        "bpawn_bb": bpawn_bb,
        "between_bb": between_bb,
        "plane_allowed_bb": plane_allowed_bb,
        "pin_movable_bb": pin_movable_bb,
        "on_board_bb": on_board_bb,
        "real_to_tbl": real_to_tbl,
        "bb_scalars": BB_SCALARS,
    }
    _LEGAL_TABLES_CACHE[device] = cached
    return cached


if _HAS_TRITON:

    @triton.jit
    def _legal_mask_kernel(
        pieces_ptr,        # int8 [B, 64]
        side_ptr,          # int8 [B]
        cr_ptr,            # int8 [B]
        ep_ptr,            # int8 [B]
        out_ptr,           # int8 [B, 64, PAD]   (we slice 73 cols in Python)
        df_tbl_ptr,        # int32 [64, PAD]
        dr_tbl_ptr,
        kind_tbl_ptr,
        ray_dir_tbl_ptr,
        ray_dist_tbl_ptr,
        promo_tbl_ptr,
        KNIGHT_TBL_ptr,    # int32 [64, 64]
        KING_TBL_ptr,
        WPAWN_TBL_ptr,
        BPAWN_TBL_ptr,
        KNIGHT_BB_ptr,     # int64 [64]  per-square attack bitboard
        KING_BB_ptr,
        WPAWN_BB_ptr,
        BPAWN_BB_ptr,
        BETWEEN_BB_ptr,    # int64 [2, 64, PAD]  (white, black)
        PLANE_ALLOWED_BB_ptr,  # int64 [7, 2]   per piece-type
        PIN_MOVABLE_BB_ptr,    # int64 [2, 9, 2] per side per pin-state
        ON_BOARD_BB_ptr,       # int64 [2, 64, 2] per side per from-sq
        REAL_TO_TBL_ptr,       # int8  [2, 64, PAD] per side per (from, plane)
        BB_IS_Q_LO: tl.constexpr,   BB_IS_Q_HI: tl.constexpr,
        BB_IS_U_LO: tl.constexpr,   BB_IS_U_HI: tl.constexpr,
        BB_IS_CASTLE_LO: tl.constexpr, BB_IS_CASTLE_HI: tl.constexpr,
        BB_PUSH_LO: tl.constexpr,   BB_PUSH_HI: tl.constexpr,
        BB_CAP_LO: tl.constexpr,    BB_CAP_HI: tl.constexpr,
        BB_PUSH1_LO: tl.constexpr,  BB_PUSH1_HI: tl.constexpr,
        BB_PUSH2_LO: tl.constexpr,  BB_PUSH2_HI: tl.constexpr,
        BB_UNDERPROMO_PUSH_LO: tl.constexpr, BB_UNDERPROMO_PUSH_HI: tl.constexpr,
        BB_PLANE_VALID_LO: tl.constexpr, BB_PLANE_VALID_HI: tl.constexpr,
        BLOCK_SQ: tl.constexpr,    # 64
        BLOCK_PL: tl.constexpr,    # 128 (PAD = output stride)
        BLOCK_PL_CHUNK: tl.constexpr,  # 64 (planes processed per chunk pass)
        NUM_PL: tl.constexpr,      # 73
    ):
        pid = tl.program_id(axis=0)
        sq = tl.arange(0, BLOCK_SQ)              # [64]

        # ----- Phase 1: load state -----
        pieces = tl.load(pieces_ptr + pid * BLOCK_SQ + sq).to(tl.int32)  # [64]
        side = tl.load(side_ptr + pid).to(tl.int32)
        cr = tl.load(cr_ptr + pid).to(tl.int32)
        ep = tl.load(ep_ptr + pid).to(tl.int32)
        is_white_mover = side == 0

        # Find mover's king square.
        king_code = tl.where(is_white_mover, 6, 12)
        king_mask_v = (pieces == king_code).to(tl.int32)
        king_sq = tl.sum(sq * king_mask_v)
        king_file = king_sq & 7
        king_rank = king_sq >> 3

        # ----- Phase 2: enemy attacks with mover king removed -----
        pieces_nk = tl.where(pieces == king_code, 0, pieces)
        pawn_code = tl.where(is_white_mover, 7, 1)
        knight_code = tl.where(is_white_mover, 8, 2)
        bishop_code = tl.where(is_white_mover, 9, 3)
        rook_code = tl.where(is_white_mover, 10, 4)
        queen_code = tl.where(is_white_mover, 11, 5)
        enemy_king_code = tl.where(is_white_mover, 12, 6)
        own_king_code = tl.where(is_white_mover, 6, 12)

        # Idea #1: per-source-square u64 attack bitboards. The matmul becomes
        # an OR-reduction over masked rows.
        KNIGHT_BB = tl.load(KNIGHT_BB_ptr + sq)         # [64] int64
        KING_BB = tl.load(KING_BB_ptr + sq)
        WP_BB = tl.load(WPAWN_BB_ptr + sq)
        BP_BB = tl.load(BPAWN_BB_ptr + sq)
        PAWN_BB = tl.where(is_white_mover, BP_BB, WP_BB)

        enemy_pawn_v = pieces_nk == pawn_code
        enemy_knight_v = pieces_nk == knight_code
        enemy_king_v = pieces_nk == enemy_king_code

        zero64 = tl.zeros([BLOCK_SQ], tl.int64)
        knight_atk_bb = tl.reduce_or(
            tl.where(enemy_knight_v, KNIGHT_BB, zero64), axis=0
        )
        king_atk_bb = tl.reduce_or(
            tl.where(enemy_king_v, KING_BB, zero64), axis=0
        )
        pawn_atk_bb = tl.reduce_or(
            tl.where(enemy_pawn_v, PAWN_BB, zero64), axis=0
        )

        # Keep these int32 [64] forms for the silenced-unused-vars block below.
        enemy_pawn = enemy_pawn_v.to(tl.int32)
        enemy_knight = enemy_knight_v.to(tl.int32)
        enemy_king = enemy_king_v.to(tl.int32)

        file_sq = sq & 7
        rank_sq = sq >> 3

        slider_atk = tl.zeros([BLOCK_SQ], tl.int32)
        DFS_C = [0, 1, 1, 1, 0, -1, -1, -1]
        DRS_C = [1, 1, 0, -1, -1, -1, 0, 1]
        ORTH_C = [True, False, True, False, True, False, True, False]

        # Vectorize across the 7 ray steps: per direction, build a [64, 8] tile
        # of target pieces (8 = next pow2 above 7; lane 0 is treated as a
        # no-op step). Replaces 7 separate gathers per direction.
        step_arr = tl.arange(0, 8)[None, :]                     # [1, 8] step 0..7
        valid_step = step_arr >= 1                              # [1, 8]
        for d_idx in tl.static_range(0, 8):
            df_d = DFS_C[d_idx]
            dr_d = DRS_C[d_idx]
            is_orth = ORTH_C[d_idx]
            tf = file_sq[:, None] + df_d * step_arr             # [64, 8]
            tr = rank_sq[:, None] + dr_d * step_arr             # [64, 8]
            on_b = ((tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8) & valid_step)
            tsq = tl.where(on_b, tr * 8 + tf, 0)                # [64, 8]
            tp = tl.gather(
                pieces_nk[:, None] + tl.zeros([BLOCK_SQ, 8], tl.int32),  # [64, 8] broadcast
                tsq, axis=0
            )
            tp = tl.where(on_b, tp, 0)
            is_piece = (tp != 0).to(tl.int32)
            cum = tl.cumsum(is_piece, axis=1)                   # [64, 8]
            first_step_mask = (cum == 1) & (is_piece == 1)
            if is_orth:
                is_match = (tp == rook_code) | (tp == queen_code)
            else:
                is_match = (tp == bishop_code) | (tp == queen_code)
            contrib = (first_step_mask & is_match).to(tl.int32).sum(axis=1)
            contrib = (contrib > 0).to(tl.int32)
            slider_atk = slider_atk | contrib

        # Pack slider [64] int32 vector into u64; combine with already-bb jump attacks.
        bit_per_sq = (tl.full([1], 1, tl.int64) << sq.to(tl.int64)).to(tl.int64)
        slider_atk_bb = tl.sum(tl.where(slider_atk != 0, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
        ea_bb = knight_atk_bb | king_atk_bb | pawn_atk_bb | slider_atk_bb

        # ----- Phase 3: ray analysis from king -----
        # For each direction d in 0..7, walk 7 steps from king. Track first/second
        # occupied square along the ray. Build:
        #   block_or_capture[64] (squares that, when targeted, resolve a check)
        #   pin_legal_mat[64, 64] (per from_sq, allowed to_sq mask)
        #   num_checkers (slider component, scalar)
        #
        # Compact pin descriptor: per-from-sq unit-vector direction of the pin
        # (or sentinel 99). Replaces the [64,64] pin_legal_mat for ~16KB shmem
        # savings.  pin_df_per_from[from] in {-1, 0, +1, 99}; same for pin_dr.
        pin_df_per_from = tl.full([BLOCK_SQ], 99, tl.int32)
        pin_dr_per_from = tl.full([BLOCK_SQ], 99, tl.int32)
        boc_bb = tl.zeros([], tl.int64)
        num_checkers_slider = 0

        # Idea #5: vectorize the per-direction 7-step ray walk as an [8] tile.
        # Replaces 56 sequential scalar steps with 8 parallel gathers.
        step8 = tl.arange(0, 8)                        # [8]
        one64_8 = tl.full([8], 1, tl.int64)
        for d_idx in tl.static_range(0, 8):
            df_d = DFS_C[d_idx]
            dr_d = DRS_C[d_idx]
            is_orth = ORTH_C[d_idx]

            tf = king_file + df_d * step8              # [8]
            tr = king_rank + dr_d * step8              # [8]
            on_b = ((tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8) & (step8 >= 1))
            tsq = tl.where(on_b, tr * 8 + tf, 0)       # [8]
            tp = tl.gather(pieces, tsq, axis=0)        # [8] int32
            tp = tl.where(on_b, tp, 0)
            is_piece = tp != 0
            cum = tl.cumsum(is_piece.to(tl.int32), axis=0)   # [8]
            first_mask = (cum == 1) & is_piece
            second_mask = (cum == 2) & is_piece
            first_p = tl.sum(tl.where(first_mask, tp, 0))
            first_sq = tl.sum(tl.where(first_mask, tsq, 0))
            second_p = tl.sum(tl.where(second_mask, tp, 0))
            has_first = tl.sum(first_mask.to(tl.int32)) > 0
            has_second = tl.sum(second_mask.to(tl.int32)) > 0

            first_is_white = (first_p >= 1) & (first_p <= 6)
            first_is_black = (first_p >= 7) & (first_p <= 12)
            first_is_own = tl.where(is_white_mover, first_is_white, first_is_black)
            first_is_enemy = tl.where(is_white_mover, first_is_black, first_is_white)
            second_is_white = (second_p >= 1) & (second_p <= 6)
            second_is_black = (second_p >= 7) & (second_p <= 12)
            second_is_enemy = tl.where(is_white_mover, second_is_black, second_is_white)

            if is_orth:
                first_is_enemy_slider = first_is_enemy & ((first_p == rook_code) | (first_p == queen_code))
                second_is_enemy_slider = second_is_enemy & ((second_p == rook_code) | (second_p == queen_code))
            else:
                first_is_enemy_slider = first_is_enemy & ((first_p == bishop_code) | (first_p == queen_code))
                second_is_enemy_slider = second_is_enemy & ((second_p == bishop_code) | (second_p == queen_code))

            is_check_ray = first_is_enemy_slider & has_first
            is_pin_ray = first_is_own & second_is_enemy_slider & has_second
            num_checkers_slider = num_checkers_slider + tl.where(is_check_ray, 1, 0)

            # block_or_capture for this dir: squares with cum<=1 & on_b (path
            # to and including the first piece). Pack directly to a u64.
            in_seg = (cum <= 1) & on_b
            ray_bb = tl.sum(tl.where(in_seg, one64_8 << tsq.to(tl.int64), tl.zeros([8], tl.int64)))
            boc_bb = boc_bb | tl.where(is_check_ray, ray_bb, tl.zeros([], tl.int64))

            df_unit_d = (1 if df_d > 0 else (-1 if df_d < 0 else 0))
            dr_unit_d = (1 if dr_d > 0 else (-1 if dr_d < 0 else 0))
            pin_match_lane = (sq == first_sq) & is_pin_ray
            pin_df_per_from = tl.where(pin_match_lane, df_unit_d, pin_df_per_from)
            pin_dr_per_from = tl.where(pin_match_lane, dr_unit_d, pin_dr_per_from)

        # Knight + pawn checkers (single-square contributions to boc_bb).
        # By knight-attack symmetry, KNIGHT_BB[king_sq] = squares attacking the
        # king. Same trick for pawn attacks: a black pawn attacks `s` to king_sq
        # iff a white pawn on king_sq would attack `s`, so WPAWN_BB[king_sq]
        # gives the squares from which an enemy black pawn checks the king.
        knight_atks_to_king_bb = tl.load(KNIGHT_BB_ptr + king_sq)
        wp_at_king_bb = tl.load(WPAWN_BB_ptr + king_sq)
        bp_at_king_bb = tl.load(BPAWN_BB_ptr + king_sq)
        pawn_atks_to_king_bb = tl.where(is_white_mover, wp_at_king_bb, bp_at_king_bb)
        enemy_knight_bb = tl.sum(
            tl.where(pieces == knight_code, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64))
        )
        enemy_pawn_bb = tl.sum(
            tl.where(pieces == pawn_code, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64))
        )
        knight_check_bb = knight_atks_to_king_bb & enemy_knight_bb
        pawn_check_bb = pawn_atks_to_king_bb & enemy_pawn_bb
        # Per-lane vectors are still needed only for num_checkers counts.
        knight_checkers = (((knight_atks_to_king_bb >> sq.to(tl.int64)) & 1)
                           & (pieces == knight_code).to(tl.int64)).to(tl.int32)
        pawn_checkers = (((pawn_atks_to_king_bb >> sq.to(tl.int64)) & 1)
                         & (pieces == pawn_code).to(tl.int64)).to(tl.int32)

        # OR knight/pawn check contributions into boc_bb (already in u64 form).
        boc_bb = boc_bb | knight_check_bb | pawn_check_bb
        num_checkers = (
            num_checkers_slider
            + tl.sum(knight_checkers)
            + tl.sum(pawn_checkers)
        )
        in_check = num_checkers >= 1
        double_check = num_checkers >= 2

        # ----- Phase 4b: per-board scalar prep for castling and EP-pin -----
        # Hoist occ_bb here so it's available for both castling and EP scans.
        occ_bb = tl.sum(tl.where(pieces != 0, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
        # Enemy occupancy bitboard — used in the per-cell phase to test
        # `to_is_enemy` via a scalar bb shift instead of a [64,64] gather.
        is_enemy_v = tl.where(
            is_white_mover,
            (pieces >= 7) & (pieces <= 12),
            (pieces >= 1) & (pieces <= 6),
        )
        enemy_bb = tl.sum(tl.where(is_enemy_v, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))

        # Castling check via bitboards. Constant masks for each of 4 castle rights:
        #   *_EMPTY: squares that must be empty (between king and rook).
        #   *_SAFE:  squares the king must not be attacked on (incl king's start).
        #   *_KSQ:   king's start bit / *_RSQ: rook's start bit.
        # Then the test is: ((occ_bb & EMPTY) == 0) & ((ea_bb & SAFE) == 0)
        #                   & ((wk_bb >> KSQ) & 1) & ((wr_bb >> RSQ) & 1).
        # Squares: e1=4, f1=5, g1=6, d1=3, c1=2, b1=1, a1=0, h1=7;
        #          e8=60, f8=61, g8=62, d8=59, c8=58, b8=57, a8=56, h8=63.
        WK_EMPTY = (1 << 5) | (1 << 6)                 # f1 g1
        WK_SAFE  = (1 << 4) | (1 << 5) | (1 << 6)      # e1 f1 g1
        WQ_EMPTY = (1 << 1) | (1 << 2) | (1 << 3)      # b1 c1 d1
        WQ_SAFE  = (1 << 2) | (1 << 3) | (1 << 4)      # c1 d1 e1
        BK_EMPTY = (1 << 61) | (1 << 62)               # f8 g8
        BK_SAFE  = (1 << 60) | (1 << 61) | (1 << 62)   # e8 f8 g8
        BQ_EMPTY = (1 << 57) | (1 << 58) | (1 << 59)   # b8 c8 d8
        BQ_SAFE  = (1 << 58) | (1 << 59) | (1 << 60)   # c8 d8 e8

        # Piece-type bitboards (white king/rook, black king/rook).
        wk_bb = tl.sum(tl.where(pieces == 6,  bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
        wr_bb = tl.sum(tl.where(pieces == 4,  bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
        bk_bb = tl.sum(tl.where(pieces == 12, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
        br_bb = tl.sum(tl.where(pieces == 10, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))

        side_white = is_white_mover
        side_black = ~is_white_mover
        cr_wk = (cr & 1) != 0
        cr_wq = ((cr >> 1) & 1) != 0
        cr_bk = ((cr >> 2) & 1) != 0
        cr_bq = ((cr >> 3) & 1) != 0

        wk_ok = (
            side_white & cr_wk
            & (((wk_bb >> 4) & 1) != 0) & (((wr_bb >> 7) & 1) != 0)
            & ((occ_bb & WK_EMPTY) == 0) & ((ea_bb & WK_SAFE) == 0)
        )
        wq_ok = (
            side_white & cr_wq
            & (((wk_bb >> 4) & 1) != 0) & (((wr_bb >> 0) & 1) != 0)
            & ((occ_bb & WQ_EMPTY) == 0) & ((ea_bb & WQ_SAFE) == 0)
        )
        bk_ok = (
            side_black & cr_bk
            & (((bk_bb >> 60) & 1) != 0) & (((br_bb >> 63) & 1) != 0)
            & ((occ_bb & BK_EMPTY) == 0) & ((ea_bb & BK_SAFE) == 0)
        )
        bq_ok = (
            side_black & cr_bq
            & (((bk_bb >> 60) & 1) != 0) & (((br_bb >> 56) & 1) != 0)
            & ((occ_bb & BQ_EMPTY) == 0) & ((ea_bb & BQ_SAFE) == 0)
        )

        # Square indices used elsewhere (EP-pin code below).
        e1, f1c, g1c, d1c, c1c, b1c, a1c, h1c = 4, 5, 6, 3, 2, 1, 0, 7
        e8, f8c, g8c, d8c, c8c, b8c, a8c, h8c = 60, 61, 62, 59, 58, 57, 56, 63

        # ---- EP horizontal-pin: per-from ep_unsafe[64] (state-dependent) ----
        # Idea #4: skip the [64,64] h-rank tile entirely when ep < 0.
        ep_valid = ep >= 0
        ep_unsafe_per_from = tl.zeros([BLOCK_SQ], tl.int32)
        if ep_valid:
            ep_cap_sq = tl.where(is_white_mover, ep - 8, ep + 8)
            ep_cap_rank = ep_cap_sq >> 3
            king_on_eprank = king_rank == ep_cap_rank
            enemy_rq_v = ((pieces == rook_code) | (pieces == queen_code)).to(tl.int32)
            from_lane = tl.arange(0, BLOCK_SQ)[:, None]
            sq_lane = tl.arange(0, BLOCK_SQ)[None, :]
            rank_of_sq = (sq_lane >> 3).to(tl.int32)
            file_of_sq_2d = (sq_lane & 7).to(tl.int32)
            on_king_rank2 = rank_of_sq == king_rank
            vacated2 = (sq_lane == from_lane) | (sq_lane == ep_cap_sq) | (sq_lane == king_sq)
            occ2 = ((pieces[None, :] != 0) & on_king_rank2 & ~vacated2).to(tl.int32)
            west_occ = occ2 & (file_of_sq_2d < king_file).to(tl.int32)
            east_occ = occ2 & (file_of_sq_2d > king_file).to(tl.int32)
            west_file_v = tl.where(west_occ > 0, file_of_sq_2d, -1)
            west_first_file = tl.max(west_file_v, axis=1)
            east_file_v = tl.where(east_occ > 0, file_of_sq_2d, 8)
            east_first_file = tl.min(east_file_v, axis=1)
            has_west = west_first_file >= 0
            has_east = east_first_file < 8
            west_sq = king_rank * 8 + tl.maximum(west_first_file, 0)
            east_sq = king_rank * 8 + tl.minimum(east_first_file, 7)
            west_enemy_rq = tl.gather(enemy_rq_v, west_sq, axis=0)
            east_enemy_rq = tl.gather(enemy_rq_v, east_sq, axis=0)
            ep_unsafe_per_from = (
                (((west_enemy_rq != 0) & has_west) | ((east_enemy_rq != 0) & has_east))
                & king_on_eprank
            ).to(tl.int32)

        # Common per-from-sq quantities for the chunk loop.
        rows2d_chunk = tl.arange(0, BLOCK_SQ)[:, None]
        forward_scalar = tl.where(is_white_mover, 8, -8)
        inter_sq_1d = tl.minimum(tl.maximum(sq + forward_scalar, 0), 63)
        inter_piece_1d = tl.load(pieces_ptr + pid * BLOCK_SQ + inter_sq_1d).to(tl.int32)
        from_piece_b = pieces[:, None]
        from_is_white_per = (from_piece_b >= 1) & (from_piece_b <= 6)
        from_is_black_per = (from_piece_b >= 7) & (from_piece_b <= 12)
        from_belongs_mover_per = tl.where(is_white_mover, from_is_white_per, from_is_black_per)
        from_is_king_per = from_piece_b == own_king_code
        pt_per = tl.where(from_is_black_per, from_piece_b - 6,
                          tl.where(from_is_white_per, from_piece_b, 0))
        is_pawn_per = pt_per == 1
        # pt for plane-allowed lookup: 0 when piece doesn't belong to mover.
        pt_for_lookup = tl.where(from_belongs_mover_per, pt_per, 0)
        rrank_from = (rows2d_chunk >> 3).to(tl.int32)
        from_rank_mover = tl.where(is_white_mover, rrank_from, 7 - rrank_from)
        mover_pawn_code = tl.where(is_white_mover, 1, 7)

        # ---- Per-from-sq plane bitmasks (lo, hi). ----
        # plane-allowed by piece type, looked up via pt_for_lookup [64].
        pt_lookup_1d = pt_for_lookup.reshape([BLOCK_SQ])
        plane_allowed_lo_per = tl.load(PLANE_ALLOWED_BB_ptr + pt_lookup_1d * 2 + 0)
        plane_allowed_hi_per = tl.load(PLANE_ALLOWED_BB_ptr + pt_lookup_1d * 2 + 1)

        # On-board bb per (side, from_sq).
        side_b_idx = tl.where(is_white_mover, 0, 1)
        on_board_base = side_b_idx * 64 * 2
        sq_long = tl.arange(0, BLOCK_SQ)
        on_board_lo_per = tl.load(ON_BOARD_BB_ptr + on_board_base + sq_long * 2 + 0)
        on_board_hi_per = tl.load(ON_BOARD_BB_ptr + on_board_base + sq_long * 2 + 1)

        # Pin-movable bb per from_sq. Compute pin_idx (0..7 = direction, 8 = unpinned).
        pin_idx = tl.where(pin_df_per_from == 99, 8,
                  tl.where((pin_df_per_from == 0)  & (pin_dr_per_from == 1),  0,   # N
                  tl.where((pin_df_per_from == 1)  & (pin_dr_per_from == 1),  1,   # NE
                  tl.where((pin_df_per_from == 1)  & (pin_dr_per_from == 0),  2,   # E
                  tl.where((pin_df_per_from == 1)  & (pin_dr_per_from == -1), 3,   # SE
                  tl.where((pin_df_per_from == 0)  & (pin_dr_per_from == -1), 4,   # S
                  tl.where((pin_df_per_from == -1) & (pin_dr_per_from == -1), 5,   # SW
                  tl.where((pin_df_per_from == -1) & (pin_dr_per_from == 0),  6,   # W
                  tl.where((pin_df_per_from == -1) & (pin_dr_per_from == 1),  7,   # NW
                                                                              8))))))))) # fallback
        pin_base = side_b_idx * 9 * 2
        pin_movable_lo_per = tl.load(PIN_MOVABLE_BB_ptr + pin_base + pin_idx * 2 + 0)
        pin_movable_hi_per = tl.load(PIN_MOVABLE_BB_ptr + pin_base + pin_idx * 2 + 1)

        ep_unsafe_b = ep_unsafe_per_from[:, None]

        # Side offset for the [2, 64, PAD] tables (between_bb, real_to_tbl).
        side_off = tl.where(is_white_mover, 0, 64 * BLOCK_PL)
        side_off_int8 = tl.where(is_white_mover, 0, 64 * BLOCK_PL)

        # Plane-index constants for castling re-enable.
        PLANE_CE = 2 * 7 + 1   # 15
        PLANE_CW = 6 * 7 + 1   # 43

        # ----- Phase 5+6+7: per-(from_sq, plane) work, in 2 chunks of 64 planes -----
        for chunk_idx in tl.static_range(0, BLOCK_PL // BLOCK_PL_CHUNK):
            plane_off = chunk_idx * BLOCK_PL_CHUNK
            local_pl = tl.arange(0, BLOCK_PL_CHUNK)         # [BLOCK_PL_CHUNK]
            local_pl_2d = local_pl[None, :]                 # [1, BLOCK_PL_CHUNK]
            local_pl_64 = local_pl.to(tl.int64)
            cols2d = local_pl_2d + plane_off                # absolute plane idx
            flat_idx = rows2d_chunk * BLOCK_PL + cols2d

            # Select per-chunk lo/hi half of bb constants.
            if chunk_idx == 0:
                allowed_per = plane_allowed_lo_per
                onboard_per = on_board_lo_per
                pin_mov_per = pin_movable_lo_per
                bb_is_q   = BB_IS_Q_LO
                bb_is_u   = BB_IS_U_LO
                bb_castle = BB_IS_CASTLE_LO
                bb_push   = BB_PUSH_LO
                bb_cap    = BB_CAP_LO
                bb_push1  = BB_PUSH1_LO
                bb_push2  = BB_PUSH2_LO
                bb_uppush = BB_UNDERPROMO_PUSH_LO
                bb_pvalid = BB_PLANE_VALID_LO
            else:
                allowed_per = plane_allowed_hi_per
                onboard_per = on_board_hi_per
                pin_mov_per = pin_movable_hi_per
                bb_is_q   = BB_IS_Q_HI
                bb_is_u   = BB_IS_U_HI
                bb_castle = BB_IS_CASTLE_HI
                bb_push   = BB_PUSH_HI
                bb_cap    = BB_CAP_HI
                bb_push1  = BB_PUSH1_HI
                bb_push2  = BB_PUSH2_HI
                bb_uppush = BB_UNDERPROMO_PUSH_HI
                bb_pvalid = BB_PLANE_VALID_HI

            # Per-cell predicates from per-from-sq u64s + per-plane bit-pos.
            allowed_bool = ((allowed_per[:, None] >> local_pl_64[None, :]) & 1) != 0
            on_board2d = ((onboard_per[:, None] >> local_pl_64[None, :]) & 1) != 0
            pin_at_ft = ((pin_mov_per[:, None] >> local_pl_64[None, :]) & 1) != 0

            # Per-plane scalar bb -> per-cell bool (broadcast across rows).
            is_q   = ((bb_is_q   >> local_pl_64) & 1) != 0   # [BLOCK_PL_CHUNK]
            is_u   = ((bb_is_u   >> local_pl_64) & 1) != 0
            is_castle_plane = ((bb_castle >> local_pl_64) & 1) != 0
            push_planes = ((bb_push >> local_pl_64) & 1) != 0
            cap_planes  = ((bb_cap  >> local_pl_64) & 1) != 0
            is_push1 = ((bb_push1 >> local_pl_64) & 1) != 0
            is_push2 = ((bb_push2 >> local_pl_64) & 1) != 0
            plane_valid_chunk = ((bb_pvalid >> local_pl_64) & 1) != 0

            # real_to from precomputed table.
            real_to = tl.load(REAL_TO_TBL_ptr + side_off_int8 + flat_idx).to(tl.int32)
            real_to_64 = real_to.to(tl.int64)

            # to_is_enemy / to_is_empty via scalar bb shifts instead of a
            # [64,64] gather over `pieces`.
            occ_at_to = ((occ_bb >> real_to_64) & 1) != 0
            to_is_empty = ~occ_at_to                       # off-board lanes are
            # gated by `on_board2d` later, so the off-board interpretation
            # doesn't matter.
            to_is_enemy = ((enemy_bb >> real_to_64) & 1) != 0

            # Slider intermediate-blocker test (precomputed bitboards).
            between_idx = side_off + rows2d_chunk * BLOCK_PL + cols2d
            between_mask = tl.load(BETWEEN_BB_ptr + between_idx)
            has_intermediate = (between_mask & occ_bb) != 0

            # Compose pseudo-legal base.
            base = (
                from_belongs_mover_per & allowed_bool & on_board2d
                & (to_is_empty | to_is_enemy)
            )
            base = base & ~(is_q[None, :] & has_intermediate)

            # Pawn-specific filtering.
            base = base & ~(is_pawn_per & push_planes[None, :] & ~to_is_empty)
            is_ep_target_cell = (ep >= 0) & (real_to == ep)
            base = base & ~(is_pawn_per & cap_planes[None, :] & ~(to_is_enemy | is_ep_target_cell))

            bad_push2_rank = is_pawn_per & is_push2[None, :] & (from_rank_mover != 1)
            base = base & ~bad_push2_rank
            inter_piece = inter_piece_1d[:, None] + tl.zeros([BLOCK_SQ, BLOCK_PL_CHUNK], tl.int32)
            bad_push2_blocked = is_pawn_per & is_push2[None, :] & (inter_piece != 0)
            base = base & ~bad_push2_blocked

            # Underpromo destination must be mover-frame rank 7 (= rank 7 white,
            # rank 0 black). Enforce by checking real_to.
            to_rank_mover = tl.where(is_white_mover, real_to >> 3, 7 - (real_to >> 3))
            bad_underpromo = is_pawn_per & is_u[None, :] & (to_rank_mover != 7) & on_board2d
            base = base & ~bad_underpromo

            # Disable naive king dist-2 moves; legal castles re-added below.
            base = base & ~(from_is_king_per & is_castle_plane[None, :])

            # Legality filter (king vs non-king).
            ea_at_to = ((ea_bb  >> real_to_64) & 1) != 0
            boc_at_to = ((boc_bb >> real_to_64) & 1) != 0
            king_legal = ~ea_at_to
            non_king_legal = (~double_check) & pin_at_ft & (~in_check | boc_at_to)
            legal_flag = tl.where(from_is_king_per, king_legal, non_king_legal)
            out_mask = base & legal_flag & plane_valid_chunk[None, :]

            # Castling re-enable for the specific castle planes.
            is_ce = (rows2d_chunk == 4) & (cols2d == PLANE_CE) & is_white_mover
            is_cw = (rows2d_chunk == 4) & (cols2d == PLANE_CW) & is_white_mover
            is_ce_b = (rows2d_chunk == 60) & (cols2d == PLANE_CE) & (~is_white_mover)
            is_cw_b = (rows2d_chunk == 60) & (cols2d == PLANE_CW) & (~is_white_mover)
            out_mask = out_mask | (is_ce & wk_ok)
            out_mask = out_mask | (is_cw & wq_ok)
            out_mask = out_mask | (is_ce_b & bk_ok)
            out_mask = out_mask | (is_cw_b & bq_ok)

            # EP filter.
            is_ep_move = (
                (out_mask != 0)
                & (real_to == ep) & (ep >= 0) & on_board2d
                & (from_piece_b == mover_pawn_code)
            )
            out_mask = out_mask & ~(is_ep_move & (ep_unsafe_b != 0))

            # Write this chunk.
            out_offset = pid * BLOCK_SQ * BLOCK_PL + rows2d_chunk * BLOCK_PL + cols2d
            tl.store(out_ptr + out_offset, out_mask.to(tl.int8))


def triton_legal_action_mask(vs: VState) -> Tensor:
    """Compute [B, ACTION_SIZE] bool legal-action mask via a single Triton
    kernel launch. Equivalent to :func:`vectorized.legal_action_mask`.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is not available")
    if vs.device.type != "cuda":
        raise RuntimeError("triton_legal_action_mask requires CUDA tensors")

    device = vs.device
    B = vs.batch_size
    tbls = _legal_tables_on(device)

    pieces = vs.pieces.contiguous()
    side = vs.side_to_move.contiguous()
    cr = vs.castling.contiguous()
    ep = vs.en_passant.contiguous()

    out = torch.empty(B, 64, _PADDED_PLANES, dtype=torch.int8, device=device)
    bb = tbls["bb_scalars"]
    _legal_mask_kernel[(B,)](
        pieces, side, cr, ep, out,
        tbls["df"], tbls["dr"], tbls["kind"], tbls["ray_dir"], tbls["ray_dist"], tbls["promo"],
        tbls["knight"], tbls["king"], tbls["wpawn"], tbls["bpawn"],
        tbls["knight_bb"], tbls["king_bb"], tbls["wpawn_bb"], tbls["bpawn_bb"],
        tbls["between_bb"],
        tbls["plane_allowed_bb"], tbls["pin_movable_bb"], tbls["on_board_bb"],
        tbls["real_to_tbl"],
        BB_IS_Q_LO=bb["is_q"][0], BB_IS_Q_HI=bb["is_q"][1],
        BB_IS_U_LO=bb["is_u"][0], BB_IS_U_HI=bb["is_u"][1],
        BB_IS_CASTLE_LO=bb["is_castle"][0], BB_IS_CASTLE_HI=bb["is_castle"][1],
        BB_PUSH_LO=bb["push"][0], BB_PUSH_HI=bb["push"][1],
        BB_CAP_LO=bb["cap"][0], BB_CAP_HI=bb["cap"][1],
        BB_PUSH1_LO=bb["push1"][0], BB_PUSH1_HI=bb["push1"][1],
        BB_PUSH2_LO=bb["push2"][0], BB_PUSH2_HI=bb["push2"][1],
        BB_UNDERPROMO_PUSH_LO=bb["underpromo_push"][0],
        BB_UNDERPROMO_PUSH_HI=bb["underpromo_push"][1],
        BB_PLANE_VALID_LO=bb["plane_valid"][0],
        BB_PLANE_VALID_HI=bb["plane_valid"][1],
        BLOCK_SQ=64, BLOCK_PL=_PADDED_PLANES, BLOCK_PL_CHUNK=64,
        NUM_PL=NUM_MOVE_PLANES,
        num_warps=2,
    )
    return out[:, :, :NUM_MOVE_PLANES].reshape(B, ACTION_SIZE).to(torch.bool)


def triton_slider_blockers(pieces: Tensor) -> Tensor:
    """Compute [B, 64, 8, 7] bool: cumulative-OR ray blockers per from-sq/dir.
    blockers[b, from_sq, dir, k] = True iff any of the squares at distance
    1..k+1 along `dir` from `from_sq` is occupied. Single Triton launch.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is not available")
    if pieces.device.type != "cuda":
        raise RuntimeError("triton_slider_blockers requires CUDA tensors")
    B = pieces.shape[0]
    out = torch.empty(B, 64, 8, 7, dtype=torch.uint8, device=pieces.device)
    _slider_blockers_kernel[(B,)](pieces.contiguous(), out, BLOCK_SQ=64)
    return out.to(torch.bool)


# ---------------------------------------------------------------------------
# Persistent rollout kernel: one program per env runs the entire `depth`-ply
# random-rollout loop in registers. Per ply: inline legal-mask (Phase 1-7),
# reservoir-sample one uniform legal action per chunk via tl.rand-tagged
# argmin, terminal/leaf-value bookkeeping, then inline step apply gated by a
# per-program `done` flag. Outputs root_action[B] (action sampled at ply 0)
# and leaf_value[B] (root-POV +1/0/-1 if a terminal was hit, else 0).
# ---------------------------------------------------------------------------


if _HAS_TRITON:

    @triton.jit
    def _rollout_kernel(
        pieces_in_ptr,   # int8  [B, 64]
        side_in_ptr,     # int8  [B]
        cr_in_ptr,       # int8  [B]
        ep_in_ptr,       # int8  [B]
        hmc_in_ptr,      # int16 [B]
        fmn_in_ptr,      # int16 [B]
        root_action_ptr, # int64 [B]
        leaf_value_ptr,  # float32 [B]
        # legal-mask tables (same layout/order as _legal_mask_kernel)
        df_tbl_ptr, dr_tbl_ptr, kind_tbl_ptr,
        ray_dir_tbl_ptr, ray_dist_tbl_ptr, promo_tbl_ptr,
        KNIGHT_TBL_ptr, KING_TBL_ptr, WPAWN_TBL_ptr, BPAWN_TBL_ptr,
        KNIGHT_BB_ptr, KING_BB_ptr, WPAWN_BB_ptr, BPAWN_BB_ptr,
        BETWEEN_BB_ptr,
        PLANE_ALLOWED_BB_ptr, PIN_MOVABLE_BB_ptr, ON_BOARD_BB_ptr,
        REAL_TO_TBL_ptr,
        # step tables (flat [64*73] int64) for _step_body
        STEP_DF_ptr, STEP_DR_ptr, STEP_KIND_ptr, STEP_PROMO_ptr,
        depth_runtime,           # int — runtime loop trip count
        seed_runtime,            # int — seeds tl.rand at runtime
        BB_IS_Q_LO: tl.constexpr,   BB_IS_Q_HI: tl.constexpr,
        BB_IS_U_LO: tl.constexpr,   BB_IS_U_HI: tl.constexpr,
        BB_IS_CASTLE_LO: tl.constexpr, BB_IS_CASTLE_HI: tl.constexpr,
        BB_PUSH_LO: tl.constexpr,   BB_PUSH_HI: tl.constexpr,
        BB_CAP_LO: tl.constexpr,    BB_CAP_HI: tl.constexpr,
        BB_PUSH1_LO: tl.constexpr,  BB_PUSH1_HI: tl.constexpr,
        BB_PUSH2_LO: tl.constexpr,  BB_PUSH2_HI: tl.constexpr,
        BB_UNDERPROMO_PUSH_LO: tl.constexpr, BB_UNDERPROMO_PUSH_HI: tl.constexpr,
        BB_PLANE_VALID_LO: tl.constexpr, BB_PLANE_VALID_HI: tl.constexpr,
        BLOCK_SQ: tl.constexpr,    # 64
        BLOCK_PL: tl.constexpr,    # 128
        BLOCK_PL_CHUNK: tl.constexpr,  # 64
        NUM_PL: tl.constexpr,      # 73
    ):
        pid = tl.program_id(axis=0)
        sq = tl.arange(0, BLOCK_SQ)

        # Initial state -> registers.
        pieces = tl.load(pieces_in_ptr + pid * BLOCK_SQ + sq).to(tl.int32)
        side = tl.load(side_in_ptr + pid).to(tl.int32)
        cr = tl.load(cr_in_ptr + pid).to(tl.int32)
        ep = tl.load(ep_in_ptr + pid).to(tl.int32)
        hmc = tl.load(hmc_in_ptr + pid).to(tl.int32)
        fmn = tl.load(fmn_in_ptr + pid).to(tl.int32)

        root_player = side  # captured before any apply (uniform per program)
        leaf_value = 0.0
        done = False
        root_action = 0

        # Direction tables for slider/ray walks (compile-time).
        DFS_C = [0, 1, 1, 1, 0, -1, -1, -1]
        DRS_C = [1, 1, 0, -1, -1, -1, 0, 1]
        ORTH_C = [True, False, True, False, True, False, True, False]

        # Castling planes (for re-enable in chunk 0).
        PLANE_CE = 2 * 7 + 1   # 15
        PLANE_CW = 6 * 7 + 1   # 43

        # Per-program random offset budget.
        # offsets_per_ply = 2 chunks * 64 sq * 64 plane = 8192
        OFF_PER_PLY = 8192

        for d in range(depth_runtime):
            base_off_ply = (pid.to(tl.int64) * (depth_runtime * OFF_PER_PLY)
                            + d.to(tl.int64) * OFF_PER_PLY)

            # ===================================================================
            # Phase 1: scalars derived from current registers.
            # ===================================================================
            is_white_mover = side == 0

            king_code = tl.where(is_white_mover, 6, 12)
            king_mask_v = (pieces == king_code).to(tl.int32)
            king_sq = tl.sum(sq * king_mask_v)
            king_file = king_sq & 7
            king_rank = king_sq >> 3

            # ===================================================================
            # Phase 2: enemy attacks with mover king removed.
            # ===================================================================
            pieces_nk = tl.where(pieces == king_code, 0, pieces)
            pawn_code = tl.where(is_white_mover, 7, 1)
            knight_code = tl.where(is_white_mover, 8, 2)
            bishop_code = tl.where(is_white_mover, 9, 3)
            rook_code = tl.where(is_white_mover, 10, 4)
            queen_code = tl.where(is_white_mover, 11, 5)
            enemy_king_code = tl.where(is_white_mover, 12, 6)
            own_king_code = tl.where(is_white_mover, 6, 12)

            KNIGHT_BB = tl.load(KNIGHT_BB_ptr + sq)
            KING_BB = tl.load(KING_BB_ptr + sq)
            WP_BB = tl.load(WPAWN_BB_ptr + sq)
            BP_BB = tl.load(BPAWN_BB_ptr + sq)
            PAWN_BB = tl.where(is_white_mover, BP_BB, WP_BB)

            enemy_pawn_v = pieces_nk == pawn_code
            enemy_knight_v = pieces_nk == knight_code
            enemy_king_v = pieces_nk == enemy_king_code

            zero64 = tl.zeros([BLOCK_SQ], tl.int64)
            knight_atk_bb = tl.reduce_or(
                tl.where(enemy_knight_v, KNIGHT_BB, zero64), axis=0
            )
            king_atk_bb = tl.reduce_or(
                tl.where(enemy_king_v, KING_BB, zero64), axis=0
            )
            pawn_atk_bb = tl.reduce_or(
                tl.where(enemy_pawn_v, PAWN_BB, zero64), axis=0
            )

            file_sq = sq & 7
            rank_sq = sq >> 3
            slider_atk = tl.zeros([BLOCK_SQ], tl.int32)

            step_arr = tl.arange(0, 8)[None, :]
            valid_step = step_arr >= 1
            for d_idx in tl.static_range(0, 8):
                df_d = DFS_C[d_idx]
                dr_d = DRS_C[d_idx]
                is_orth = ORTH_C[d_idx]
                tf = file_sq[:, None] + df_d * step_arr
                tr = rank_sq[:, None] + dr_d * step_arr
                on_b = ((tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8) & valid_step)
                tsq = tl.where(on_b, tr * 8 + tf, 0)
                tp = tl.gather(
                    pieces_nk[:, None] + tl.zeros([BLOCK_SQ, 8], tl.int32),
                    tsq, axis=0
                )
                tp = tl.where(on_b, tp, 0)
                is_piece = (tp != 0).to(tl.int32)
                cum = tl.cumsum(is_piece, axis=1)
                first_step_mask = (cum == 1) & (is_piece == 1)
                if is_orth:
                    is_match = (tp == rook_code) | (tp == queen_code)
                else:
                    is_match = (tp == bishop_code) | (tp == queen_code)
                contrib = (first_step_mask & is_match).to(tl.int32).sum(axis=1)
                contrib = (contrib > 0).to(tl.int32)
                slider_atk = slider_atk | contrib

            bit_per_sq = (tl.full([1], 1, tl.int64) << sq.to(tl.int64)).to(tl.int64)
            slider_atk_bb = tl.sum(
                tl.where(slider_atk != 0, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64))
            )
            ea_bb = knight_atk_bb | king_atk_bb | pawn_atk_bb | slider_atk_bb

            # ===================================================================
            # Phase 3: ray analysis from king (boc_bb, pin descriptors, checkers).
            # ===================================================================
            pin_df_per_from = tl.full([BLOCK_SQ], 99, tl.int32)
            pin_dr_per_from = tl.full([BLOCK_SQ], 99, tl.int32)
            boc_bb = tl.zeros([], tl.int64)
            num_checkers_slider = 0

            step8 = tl.arange(0, 8)
            one64_8 = tl.full([8], 1, tl.int64)
            for d_idx in tl.static_range(0, 8):
                df_d = DFS_C[d_idx]
                dr_d = DRS_C[d_idx]
                is_orth = ORTH_C[d_idx]

                tf = king_file + df_d * step8
                tr = king_rank + dr_d * step8
                on_b = ((tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8) & (step8 >= 1))
                tsq = tl.where(on_b, tr * 8 + tf, 0)
                tp = tl.gather(pieces, tsq, axis=0)
                tp = tl.where(on_b, tp, 0)
                is_piece = tp != 0
                cum = tl.cumsum(is_piece.to(tl.int32), axis=0)
                first_mask = (cum == 1) & is_piece
                second_mask = (cum == 2) & is_piece
                first_p = tl.sum(tl.where(first_mask, tp, 0))
                first_sq = tl.sum(tl.where(first_mask, tsq, 0))
                second_p = tl.sum(tl.where(second_mask, tp, 0))
                has_first = tl.sum(first_mask.to(tl.int32)) > 0
                has_second = tl.sum(second_mask.to(tl.int32)) > 0

                first_is_white = (first_p >= 1) & (first_p <= 6)
                first_is_black = (first_p >= 7) & (first_p <= 12)
                first_is_own = tl.where(is_white_mover, first_is_white, first_is_black)
                first_is_enemy = tl.where(is_white_mover, first_is_black, first_is_white)
                second_is_white = (second_p >= 1) & (second_p <= 6)
                second_is_black = (second_p >= 7) & (second_p <= 12)
                second_is_enemy = tl.where(is_white_mover, second_is_black, second_is_white)

                if is_orth:
                    first_is_enemy_slider = first_is_enemy & ((first_p == rook_code) | (first_p == queen_code))
                    second_is_enemy_slider = second_is_enemy & ((second_p == rook_code) | (second_p == queen_code))
                else:
                    first_is_enemy_slider = first_is_enemy & ((first_p == bishop_code) | (first_p == queen_code))
                    second_is_enemy_slider = second_is_enemy & ((second_p == bishop_code) | (second_p == queen_code))

                is_check_ray = first_is_enemy_slider & has_first
                is_pin_ray = first_is_own & second_is_enemy_slider & has_second
                num_checkers_slider = num_checkers_slider + tl.where(is_check_ray, 1, 0)

                in_seg = (cum <= 1) & on_b
                ray_bb = tl.sum(tl.where(in_seg, one64_8 << tsq.to(tl.int64), tl.zeros([8], tl.int64)))
                boc_bb = boc_bb | tl.where(is_check_ray, ray_bb, tl.zeros([], tl.int64))

                df_unit_d = (1 if df_d > 0 else (-1 if df_d < 0 else 0))
                dr_unit_d = (1 if dr_d > 0 else (-1 if dr_d < 0 else 0))
                pin_match_lane = (sq == first_sq) & is_pin_ray
                pin_df_per_from = tl.where(pin_match_lane, df_unit_d, pin_df_per_from)
                pin_dr_per_from = tl.where(pin_match_lane, dr_unit_d, pin_dr_per_from)

            # Knight + pawn checkers.
            knight_atks_to_king_bb = tl.load(KNIGHT_BB_ptr + king_sq)
            wp_at_king_bb = tl.load(WPAWN_BB_ptr + king_sq)
            bp_at_king_bb = tl.load(BPAWN_BB_ptr + king_sq)
            pawn_atks_to_king_bb = tl.where(is_white_mover, wp_at_king_bb, bp_at_king_bb)
            enemy_knight_bb = tl.sum(
                tl.where(pieces == knight_code, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64))
            )
            enemy_pawn_bb = tl.sum(
                tl.where(pieces == pawn_code, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64))
            )
            knight_check_bb = knight_atks_to_king_bb & enemy_knight_bb
            pawn_check_bb = pawn_atks_to_king_bb & enemy_pawn_bb
            knight_checkers = (((knight_atks_to_king_bb >> sq.to(tl.int64)) & 1)
                               & (pieces == knight_code).to(tl.int64)).to(tl.int32)
            pawn_checkers = (((pawn_atks_to_king_bb >> sq.to(tl.int64)) & 1)
                             & (pieces == pawn_code).to(tl.int64)).to(tl.int32)

            boc_bb = boc_bb | knight_check_bb | pawn_check_bb
            num_checkers = (
                num_checkers_slider
                + tl.sum(knight_checkers)
                + tl.sum(pawn_checkers)
            )
            in_check = num_checkers >= 1
            double_check = num_checkers >= 2

            # ===================================================================
            # Phase 4: occ/enemy bb, castling, EP-pin scan.
            # ===================================================================
            occ_bb = tl.sum(tl.where(pieces != 0, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
            is_enemy_v = tl.where(
                is_white_mover,
                (pieces >= 7) & (pieces <= 12),
                (pieces >= 1) & (pieces <= 6),
            )
            enemy_bb = tl.sum(tl.where(is_enemy_v, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))

            WK_EMPTY = (1 << 5) | (1 << 6)
            WK_SAFE  = (1 << 4) | (1 << 5) | (1 << 6)
            WQ_EMPTY = (1 << 1) | (1 << 2) | (1 << 3)
            WQ_SAFE  = (1 << 2) | (1 << 3) | (1 << 4)
            BK_EMPTY = (1 << 61) | (1 << 62)
            BK_SAFE  = (1 << 60) | (1 << 61) | (1 << 62)
            BQ_EMPTY = (1 << 57) | (1 << 58) | (1 << 59)
            BQ_SAFE  = (1 << 58) | (1 << 59) | (1 << 60)

            wk_bb = tl.sum(tl.where(pieces == 6,  bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
            wr_bb = tl.sum(tl.where(pieces == 4,  bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
            bk_bb = tl.sum(tl.where(pieces == 12, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
            br_bb = tl.sum(tl.where(pieces == 10, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))

            side_white = is_white_mover
            side_black = ~is_white_mover
            cr_wk = (cr & 1) != 0
            cr_wq = ((cr >> 1) & 1) != 0
            cr_bk = ((cr >> 2) & 1) != 0
            cr_bq = ((cr >> 3) & 1) != 0

            wk_ok = (
                side_white & cr_wk
                & (((wk_bb >> 4) & 1) != 0) & (((wr_bb >> 7) & 1) != 0)
                & ((occ_bb & WK_EMPTY) == 0) & ((ea_bb & WK_SAFE) == 0)
            )
            wq_ok = (
                side_white & cr_wq
                & (((wk_bb >> 4) & 1) != 0) & (((wr_bb >> 0) & 1) != 0)
                & ((occ_bb & WQ_EMPTY) == 0) & ((ea_bb & WQ_SAFE) == 0)
            )
            bk_ok = (
                side_black & cr_bk
                & (((bk_bb >> 60) & 1) != 0) & (((br_bb >> 63) & 1) != 0)
                & ((occ_bb & BK_EMPTY) == 0) & ((ea_bb & BK_SAFE) == 0)
            )
            bq_ok = (
                side_black & cr_bq
                & (((bk_bb >> 60) & 1) != 0) & (((br_bb >> 56) & 1) != 0)
                & ((occ_bb & BQ_EMPTY) == 0) & ((ea_bb & BQ_SAFE) == 0)
            )

            ep_valid = ep >= 0
            ep_unsafe_per_from = tl.zeros([BLOCK_SQ], tl.int32)
            if ep_valid:
                ep_cap_sq = tl.where(is_white_mover, ep - 8, ep + 8)
                ep_cap_rank = ep_cap_sq >> 3
                king_on_eprank = king_rank == ep_cap_rank
                enemy_rq_v = ((pieces == rook_code) | (pieces == queen_code)).to(tl.int32)
                from_lane = tl.arange(0, BLOCK_SQ)[:, None]
                sq_lane = tl.arange(0, BLOCK_SQ)[None, :]
                rank_of_sq = (sq_lane >> 3).to(tl.int32)
                file_of_sq_2d = (sq_lane & 7).to(tl.int32)
                on_king_rank2 = rank_of_sq == king_rank
                vacated2 = (sq_lane == from_lane) | (sq_lane == ep_cap_sq) | (sq_lane == king_sq)
                occ2 = ((pieces[None, :] != 0) & on_king_rank2 & ~vacated2).to(tl.int32)
                west_occ = occ2 & (file_of_sq_2d < king_file).to(tl.int32)
                east_occ = occ2 & (file_of_sq_2d > king_file).to(tl.int32)
                west_file_v = tl.where(west_occ > 0, file_of_sq_2d, -1)
                west_first_file = tl.max(west_file_v, axis=1)
                east_file_v = tl.where(east_occ > 0, file_of_sq_2d, 8)
                east_first_file = tl.min(east_file_v, axis=1)
                has_west = west_first_file >= 0
                has_east = east_first_file < 8
                west_sq = king_rank * 8 + tl.maximum(west_first_file, 0)
                east_sq = king_rank * 8 + tl.minimum(east_first_file, 7)
                west_enemy_rq = tl.gather(enemy_rq_v, west_sq, axis=0)
                east_enemy_rq = tl.gather(enemy_rq_v, east_sq, axis=0)
                ep_unsafe_per_from = (
                    (((west_enemy_rq != 0) & has_west) | ((east_enemy_rq != 0) & has_east))
                    & king_on_eprank
                ).to(tl.int32)

            # Per-from-sq common quantities for the chunk loop.
            rows2d_chunk = tl.arange(0, BLOCK_SQ)[:, None]
            forward_scalar = tl.where(is_white_mover, 8, -8)
            inter_sq_1d = tl.minimum(tl.maximum(sq + forward_scalar, 0), 63)
            # Use register gather instead of global load (pieces is in registers).
            inter_piece_1d = tl.gather(pieces, inter_sq_1d, axis=0)
            from_piece_b = pieces[:, None]
            from_is_white_per = (from_piece_b >= 1) & (from_piece_b <= 6)
            from_is_black_per = (from_piece_b >= 7) & (from_piece_b <= 12)
            from_belongs_mover_per = tl.where(is_white_mover, from_is_white_per, from_is_black_per)
            from_is_king_per = from_piece_b == own_king_code
            pt_per = tl.where(from_is_black_per, from_piece_b - 6,
                              tl.where(from_is_white_per, from_piece_b, 0))
            is_pawn_per = pt_per == 1
            pt_for_lookup = tl.where(from_belongs_mover_per, pt_per, 0)
            rrank_from = (rows2d_chunk >> 3).to(tl.int32)
            from_rank_mover = tl.where(is_white_mover, rrank_from, 7 - rrank_from)
            mover_pawn_code = tl.where(is_white_mover, 1, 7)

            pt_lookup_1d = pt_for_lookup.reshape([BLOCK_SQ])
            plane_allowed_lo_per = tl.load(PLANE_ALLOWED_BB_ptr + pt_lookup_1d * 2 + 0)
            plane_allowed_hi_per = tl.load(PLANE_ALLOWED_BB_ptr + pt_lookup_1d * 2 + 1)

            side_b_idx = tl.where(is_white_mover, 0, 1)
            on_board_base = side_b_idx * 64 * 2
            sq_long = tl.arange(0, BLOCK_SQ)
            on_board_lo_per = tl.load(ON_BOARD_BB_ptr + on_board_base + sq_long * 2 + 0)
            on_board_hi_per = tl.load(ON_BOARD_BB_ptr + on_board_base + sq_long * 2 + 1)

            pin_idx = tl.where(pin_df_per_from == 99, 8,
                      tl.where((pin_df_per_from == 0)  & (pin_dr_per_from == 1),  0,
                      tl.where((pin_df_per_from == 1)  & (pin_dr_per_from == 1),  1,
                      tl.where((pin_df_per_from == 1)  & (pin_dr_per_from == 0),  2,
                      tl.where((pin_df_per_from == 1)  & (pin_dr_per_from == -1), 3,
                      tl.where((pin_df_per_from == 0)  & (pin_dr_per_from == -1), 4,
                      tl.where((pin_df_per_from == -1) & (pin_dr_per_from == -1), 5,
                      tl.where((pin_df_per_from == -1) & (pin_dr_per_from == 0),  6,
                      tl.where((pin_df_per_from == -1) & (pin_dr_per_from == 1),  7,
                                                                                  8)))))))))
            pin_base = side_b_idx * 9 * 2
            pin_movable_lo_per = tl.load(PIN_MOVABLE_BB_ptr + pin_base + pin_idx * 2 + 0)
            pin_movable_hi_per = tl.load(PIN_MOVABLE_BB_ptr + pin_base + pin_idx * 2 + 1)

            ep_unsafe_b = ep_unsafe_per_from[:, None]

            side_off = tl.where(is_white_mover, 0, 64 * BLOCK_PL)
            side_off_int8 = tl.where(is_white_mover, 0, 64 * BLOCK_PL)

            # ===================================================================
            # Phase 5+6+7: per-chunk mask + reservoir sample.
            # ===================================================================
            running_min = tl.full([], 2.0, tl.float32)
            running_action = 0
            # 4 chunks of 32 planes (was 2 chunks of 64): halves every [64×N]
            # tile in the chunk body, reducing live register pressure.
            for chunk_idx in tl.static_range(0, BLOCK_PL // BLOCK_PL_CHUNK):
                plane_off = chunk_idx * BLOCK_PL_CHUNK
                local_pl = tl.arange(0, BLOCK_PL_CHUNK)
                local_pl_2d = local_pl[None, :]
                # Sub-offset within LO/HI half: chunks 0,2 → 0; chunks 1,3 → 32.
                sub_off = (chunk_idx % 2) * 32
                shift_amt = local_pl.to(tl.int64) + sub_off
                cols2d = local_pl_2d + plane_off

                if chunk_idx < 2:
                    allowed_per = plane_allowed_lo_per
                    onboard_per = on_board_lo_per
                    pin_mov_per = pin_movable_lo_per
                    bb_is_q   = BB_IS_Q_LO
                    bb_is_u   = BB_IS_U_LO
                    bb_castle = BB_IS_CASTLE_LO
                    bb_push   = BB_PUSH_LO
                    bb_cap    = BB_CAP_LO
                    bb_push1  = BB_PUSH1_LO
                    bb_push2  = BB_PUSH2_LO
                    bb_uppush = BB_UNDERPROMO_PUSH_LO
                    bb_pvalid = BB_PLANE_VALID_LO
                else:
                    allowed_per = plane_allowed_hi_per
                    onboard_per = on_board_hi_per
                    pin_mov_per = pin_movable_hi_per
                    bb_is_q   = BB_IS_Q_HI
                    bb_is_u   = BB_IS_U_HI
                    bb_castle = BB_IS_CASTLE_HI
                    bb_push   = BB_PUSH_HI
                    bb_cap    = BB_CAP_HI
                    bb_push1  = BB_PUSH1_HI
                    bb_push2  = BB_PUSH2_HI
                    bb_uppush = BB_UNDERPROMO_PUSH_HI
                    bb_pvalid = BB_PLANE_VALID_HI

                allowed_bool = ((allowed_per[:, None] >> shift_amt[None, :]) & 1) != 0
                on_board2d = ((onboard_per[:, None] >> shift_amt[None, :]) & 1) != 0
                pin_at_ft = ((pin_mov_per[:, None] >> shift_amt[None, :]) & 1) != 0

                is_q   = ((bb_is_q   >> shift_amt) & 1) != 0
                is_u   = ((bb_is_u   >> shift_amt) & 1) != 0
                is_castle_plane = ((bb_castle >> shift_amt) & 1) != 0
                push_planes = ((bb_push >> shift_amt) & 1) != 0
                cap_planes  = ((bb_cap  >> shift_amt) & 1) != 0
                is_push2 = ((bb_push2 >> shift_amt) & 1) != 0
                plane_valid_chunk = ((bb_pvalid >> shift_amt) & 1) != 0

                flat_idx2d = rows2d_chunk * BLOCK_PL + cols2d
                real_to = tl.load(REAL_TO_TBL_ptr + side_off_int8 + flat_idx2d).to(tl.int32)
                real_to_64 = real_to.to(tl.int64)

                occ_at_to = ((occ_bb >> real_to_64) & 1) != 0
                to_is_empty = ~occ_at_to
                to_is_enemy = ((enemy_bb >> real_to_64) & 1) != 0

                between_idx = side_off + rows2d_chunk * BLOCK_PL + cols2d
                between_mask = tl.load(BETWEEN_BB_ptr + between_idx)
                has_intermediate = (between_mask & occ_bb) != 0

                base = (
                    from_belongs_mover_per & allowed_bool & on_board2d
                    & (to_is_empty | to_is_enemy)
                )
                base = base & ~(is_q[None, :] & has_intermediate)
                base = base & ~(is_pawn_per & push_planes[None, :] & ~to_is_empty)
                is_ep_target_cell = (ep >= 0) & (real_to == ep)
                base = base & ~(is_pawn_per & cap_planes[None, :] & ~(to_is_enemy | is_ep_target_cell))

                bad_push2_rank = is_pawn_per & is_push2[None, :] & (from_rank_mover != 1)
                base = base & ~bad_push2_rank
                inter_piece = inter_piece_1d[:, None] + tl.zeros([BLOCK_SQ, BLOCK_PL_CHUNK], tl.int32)
                bad_push2_blocked = is_pawn_per & is_push2[None, :] & (inter_piece != 0)
                base = base & ~bad_push2_blocked

                to_rank_mover = tl.where(is_white_mover, real_to >> 3, 7 - (real_to >> 3))
                bad_underpromo = is_pawn_per & is_u[None, :] & (to_rank_mover != 7) & on_board2d
                base = base & ~bad_underpromo

                base = base & ~(from_is_king_per & is_castle_plane[None, :])

                ea_at_to = ((ea_bb  >> real_to_64) & 1) != 0
                boc_at_to = ((boc_bb >> real_to_64) & 1) != 0
                king_legal = ~ea_at_to
                non_king_legal = (~double_check) & pin_at_ft & (~in_check | boc_at_to)
                legal_flag = tl.where(from_is_king_per, king_legal, non_king_legal)
                out_mask = base & legal_flag & plane_valid_chunk[None, :]

                is_ce = (rows2d_chunk == 4) & (cols2d == PLANE_CE) & is_white_mover
                is_cw = (rows2d_chunk == 4) & (cols2d == PLANE_CW) & is_white_mover
                is_ce_b = (rows2d_chunk == 60) & (cols2d == PLANE_CE) & (~is_white_mover)
                is_cw_b = (rows2d_chunk == 60) & (cols2d == PLANE_CW) & (~is_white_mover)
                out_mask = out_mask | (is_ce & wk_ok)
                out_mask = out_mask | (is_cw & wq_ok)
                out_mask = out_mask | (is_ce_b & bk_ok)
                out_mask = out_mask | (is_cw_b & bq_ok)

                is_ep_move = (
                    (out_mask != 0)
                    & (real_to == ep) & (ep >= 0) & on_board2d
                    & (from_piece_b == mover_pawn_code)
                )
                out_mask = out_mask & ~(is_ep_move & (ep_unsafe_b != 0))

                # ----- Reservoir sample over this chunk's set cells -----
                # tag = U[0,1) for legal cells, sentinel 2.0 for illegal.
                # Distinct offsets per (pid, d, chunk_idx, sq, plane_local).
                offsets2d = (base_off_ply
                             + chunk_idx * (BLOCK_SQ * BLOCK_PL_CHUNK)
                             + rows2d_chunk.to(tl.int64) * BLOCK_PL_CHUNK
                             + local_pl_2d.to(tl.int64))
                tag = tl.rand(seed_runtime, offsets2d)
                sentinel_tile = tl.full([BLOCK_SQ, BLOCK_PL_CHUNK], 2.0, tl.float32)
                tag = tl.where(out_mask, tag, sentinel_tile)

                flat_tag = tag.reshape([BLOCK_SQ * BLOCK_PL_CHUNK])
                chunk_min = tl.min(flat_tag)
                # Argmin via masked-arange-min: pick the smallest flat index
                # whose value equals chunk_min (deterministic tie-break).
                idx_arr = tl.arange(0, BLOCK_SQ * BLOCK_PL_CHUNK)
                match = tl.where(flat_tag == chunk_min, idx_arr, BLOCK_SQ * BLOCK_PL_CHUNK)
                chunk_argmin = tl.min(match)
                chunk_sq = chunk_argmin // BLOCK_PL_CHUNK
                chunk_pl_local = chunk_argmin % BLOCK_PL_CHUNK
                chunk_action = chunk_sq * NUM_PL + (chunk_pl_local + chunk_idx * BLOCK_PL_CHUNK)

                improved = chunk_min < running_min
                running_min = tl.where(improved, chunk_min, running_min)
                running_action = tl.where(improved, chunk_action, running_action)

            # ----- Terminal handling (after both chunks) -----
            any_legal = running_min < 1.5
            terminal_now = (~done) & (~any_legal)
            # Mover loses iff in_check at terminal (checkmate); else stalemate=draw.
            # Root-POV: if in_check and root != mover -> +1; in_check and root==mover -> -1.
            mover_loses_pov = tl.where(root_player != side, 1.0, -1.0)
            terminal_value = tl.where(in_check, mover_loses_pov, 0.0)
            leaf_value = tl.where(terminal_now, terminal_value, leaf_value)
            done = done | terminal_now

            # ----- Capture root action at d == 0 -----
            if d == 0:
                root_action = running_action

            # ----- Apply step (gated by done; safe action 0 falls through) -----
            action = running_action
            new_p, new_s, new_cr, new_ep_, new_hmc, new_fmn = _step_body(
                pieces, side, cr, ep, hmc, fmn, action,
                STEP_DF_ptr, STEP_DR_ptr, STEP_KIND_ptr, STEP_PROMO_ptr,
                BLOCK_SQ=BLOCK_SQ, NUM_PLANES_C=NUM_PL,
            )
            pieces = tl.where(done, pieces, new_p)
            side = tl.where(done, side, new_s)
            cr = tl.where(done, cr, new_cr)
            ep = tl.where(done, ep, new_ep_)
            hmc = tl.where(done, hmc, new_hmc)
            fmn = tl.where(done, fmn, new_fmn)

        tl.store(root_action_ptr + pid, root_action.to(tl.int64))
        tl.store(leaf_value_ptr + pid, leaf_value)


def triton_rollout(
    initial_vs: VState, depth: int, seed: int = 0
) -> tuple[Tensor, Tensor]:
    """Run a full random-rollout (`depth` plies) per env in one Triton kernel.

    Per env: at each ply compute legal mask in registers, reservoir-sample one
    uniform legal action, freeze on terminal (no-move). Captures the action
    sampled at ply 0 as the "root action". leaf_value is +1/-1/0 from
    root-player POV if a terminal was reached during the rollout, else 0
    (matches the semantics of bench_mcts._run_vec_mcts).

    Returns (root_action [B] int64, leaf_value [B] float32).
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is not available")
    if initial_vs.device.type != "cuda":
        raise RuntimeError("triton_rollout requires CUDA tensors")

    device = initial_vs.device
    B = initial_vs.batch_size
    tbls = _legal_tables_on(device)
    df_t, dr_t, kd_t, pr_t = _tables_on(device)

    pieces = initial_vs.pieces.contiguous()
    side = initial_vs.side_to_move.contiguous()
    cr = initial_vs.castling.contiguous()
    ep = initial_vs.en_passant.contiguous()
    hmc = initial_vs.halfmove_clock.contiguous()
    fmn = initial_vs.fullmove_number.contiguous()

    root_action = torch.zeros(B, dtype=torch.int64, device=device)
    leaf_value = torch.zeros(B, dtype=torch.float32, device=device)

    bb = tbls["bb_scalars"]
    _rollout_kernel[(B,)](
        pieces, side, cr, ep, hmc, fmn,
        root_action, leaf_value,
        tbls["df"], tbls["dr"], tbls["kind"],
        tbls["ray_dir"], tbls["ray_dist"], tbls["promo"],
        tbls["knight"], tbls["king"], tbls["wpawn"], tbls["bpawn"],
        tbls["knight_bb"], tbls["king_bb"], tbls["wpawn_bb"], tbls["bpawn_bb"],
        tbls["between_bb"],
        tbls["plane_allowed_bb"], tbls["pin_movable_bb"], tbls["on_board_bb"],
        tbls["real_to_tbl"],
        df_t, dr_t, kd_t, pr_t,
        depth, seed,
        BB_IS_Q_LO=bb["is_q"][0], BB_IS_Q_HI=bb["is_q"][1],
        BB_IS_U_LO=bb["is_u"][0], BB_IS_U_HI=bb["is_u"][1],
        BB_IS_CASTLE_LO=bb["is_castle"][0], BB_IS_CASTLE_HI=bb["is_castle"][1],
        BB_PUSH_LO=bb["push"][0], BB_PUSH_HI=bb["push"][1],
        BB_CAP_LO=bb["cap"][0], BB_CAP_HI=bb["cap"][1],
        BB_PUSH1_LO=bb["push1"][0], BB_PUSH1_HI=bb["push1"][1],
        BB_PUSH2_LO=bb["push2"][0], BB_PUSH2_HI=bb["push2"][1],
        BB_UNDERPROMO_PUSH_LO=bb["underpromo_push"][0],
        BB_UNDERPROMO_PUSH_HI=bb["underpromo_push"][1],
        BB_PLANE_VALID_LO=bb["plane_valid"][0],
        BB_PLANE_VALID_HI=bb["plane_valid"][1],
        BLOCK_SQ=64, BLOCK_PL=_PADDED_PLANES, BLOCK_PL_CHUNK=32,
        NUM_PL=NUM_MOVE_PLANES,
        num_warps=2,
    )
    return root_action, leaf_value
