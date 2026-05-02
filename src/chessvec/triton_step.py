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

        from_sq = action // NUM_PLANES_C
        plane = action % NUM_PLANES_C
        tbl_idx = from_sq * NUM_PLANES_C + plane
        df = tl.load(df_table_ptr + tbl_idx).to(tl.int32)
        dr = tl.load(dr_table_ptr + tbl_idx).to(tl.int32)
        kind_v = tl.load(kind_table_ptr + tbl_idx).to(tl.int32)
        promo = tl.load(promo_table_ptr + tbl_idx).to(tl.int32)

        # ----- Geometry -----
        is_white = side == 0
        dr_signed = tl.where(is_white, dr, -dr)
        f0 = from_sq & 7
        r0 = from_sq // 8
        f1 = f0 + df
        r1 = r0 + dr_signed
        to_sq = r1 * 8 + f1

        # Per-env scalars from the [64]-vector via masked sum.
        moving_piece = tl.sum(tl.where(sq == from_sq, pieces, 0))
        captured_piece = tl.sum(tl.where(sq == to_sq, pieces, 0))
        pt = tl.where(moving_piece >= 7, moving_piece - 6, moving_piece)

        is_pawn = pt == 1
        is_king = pt == 6

        # En passant capture detection (pawn moves to ep with empty target).
        is_ep_capture = is_pawn & (to_sq == ep) & (captured_piece == 0) & (ep >= 0)
        ep_cap_sq = tl.where(is_white, to_sq - 8, to_sq + 8)

        # Promotion (queen by default for queen-like, else underpromo piece).
        to_rank_mover = tl.where(is_white, to_sq // 8, 7 - (to_sq // 8))
        is_promotion = is_pawn & (to_rank_mover == 7)
        promo_pt = tl.where(kind_v == 2, promo, 5)
        promoted_piece = tl.where(is_white, promo_pt, promo_pt + 6)
        new_piece = tl.where(is_promotion, promoted_piece, moving_piece)

        # Castling: relocate the rook.
        is_castle = is_king & ((df == 2) | (df == -2))
        castle_ks = is_castle & (df == 2)
        rank_b = from_sq // 8
        rook_from = tl.where(castle_ks, rank_b * 8 + 7, rank_b * 8 + 0)
        rook_to = tl.where(castle_ks, rank_b * 8 + 5, rank_b * 8 + 3)
        rook_piece = tl.where(is_white, 4, 10)  # WR or BR

        # ----- Build new pieces vector -----
        new_pieces = pieces
        # En-passant: clear the captured pawn's square.
        new_pieces = tl.where(is_ep_capture & (sq == ep_cap_sq), 0, new_pieces)
        # Clear from-sq.
        new_pieces = tl.where(sq == from_sq, 0, new_pieces)
        # Set to-sq with possibly-promoted piece.
        new_pieces = tl.where(sq == to_sq, new_piece, new_pieces)
        # Castling rook.
        new_pieces = tl.where(is_castle & (sq == rook_from), 0, new_pieces)
        new_pieces = tl.where(is_castle & (sq == rook_to), rook_piece, new_pieces)

        tl.store(out_pieces_ptr + pid * BLOCK_SQ + sq, new_pieces.to(tl.int8))

        # ----- Castling rights -----
        new_cr = cr
        wk_moved = is_king & is_white
        bk_moved = is_king & (~is_white)
        # Bits: 0=WK, 1=WQ, 2=BK, 3=BQ
        new_cr = tl.where(wk_moved, new_cr & ~0x3, new_cr)
        new_cr = tl.where(bk_moved, new_cr & ~0xC, new_cr)
        # Rook moved off / captured on starting square.
        # a1=0 -> WQ (bit1), h1=7 -> WK (bit0), a8=56 -> BQ (bit3), h8=63 -> BK (bit2).
        aff_a1 = (from_sq == 0) | (to_sq == 0)
        aff_h1 = (from_sq == 7) | (to_sq == 7)
        aff_a8 = (from_sq == 56) | (to_sq == 56)
        aff_h8 = (from_sq == 63) | (to_sq == 63)
        new_cr = tl.where(aff_a1, new_cr & ~0x2, new_cr)
        new_cr = tl.where(aff_h1, new_cr & ~0x1, new_cr)
        new_cr = tl.where(aff_a8, new_cr & ~0x8, new_cr)
        new_cr = tl.where(aff_h8, new_cr & ~0x4, new_cr)

        # ----- En passant target -----
        is_double_push = is_pawn & (df == 0) & (dr == 2)
        new_ep = tl.where(
            is_double_push,
            tl.where(is_white, to_sq - 8, to_sq + 8),
            -1,
        )

        # ----- Halfmove clock / fullmove / side -----
        is_capture = (captured_piece != 0) | is_ep_capture
        new_hmc = tl.where(is_pawn | is_capture, 0, hmc + 1)
        new_fmn = tl.where(is_white, fmn, fmn + 1)
        new_side = 1 - side

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

        rows = tl.arange(0, BLOCK_SQ)[:, None]
        cols = tl.arange(0, BLOCK_SQ)[None, :]
        KNIGHT_TBL = tl.load(KNIGHT_TBL_ptr + rows * 64 + cols).to(tl.int32)
        KING_TBL = tl.load(KING_TBL_ptr + rows * 64 + cols).to(tl.int32)
        WP_TBL = tl.load(WPAWN_TBL_ptr + rows * 64 + cols).to(tl.int32)
        BP_TBL = tl.load(BPAWN_TBL_ptr + rows * 64 + cols).to(tl.int32)
        # Mover white -> enemy black -> use BPAWN attacks (squares attacked BY black pawns).
        PAWN_TBL = tl.where(is_white_mover, BP_TBL, WP_TBL)

        enemy_pawn = (pieces_nk == pawn_code).to(tl.int32)
        enemy_knight = (pieces_nk == knight_code).to(tl.int32)
        enemy_king = (pieces_nk == enemy_king_code).to(tl.int32)

        knight_atk = (tl.sum(enemy_knight[:, None] * KNIGHT_TBL, axis=0) > 0).to(tl.int32)
        king_atk = (tl.sum(enemy_king[:, None] * KING_TBL, axis=0) > 0).to(tl.int32)
        pawn_atk = (tl.sum(enemy_pawn[:, None] * PAWN_TBL, axis=0) > 0).to(tl.int32)

        file_sq = sq & 7
        rank_sq = sq >> 3
        is_orth_slider = ((pieces_nk == rook_code) | (pieces_nk == queen_code)).to(tl.int32)
        is_diag_slider = ((pieces_nk == bishop_code) | (pieces_nk == queen_code)).to(tl.int32)
        _ = is_orth_slider + is_diag_slider  # silence unused

        slider_atk = tl.zeros([BLOCK_SQ], tl.int32)
        DFS_C = [0, 1, 1, 1, 0, -1, -1, -1]
        DRS_C = [1, 1, 0, -1, -1, -1, 0, 1]
        ORTH_C = [True, False, True, False, True, False, True, False]

        for d_idx in tl.static_range(0, 8):
            df_d = DFS_C[d_idx]
            dr_d = DRS_C[d_idx]
            is_orth = ORTH_C[d_idx]
            walking = tl.full([BLOCK_SQ], 1, tl.int32)
            for step in tl.static_range(1, 8):
                tf = file_sq + df_d * step
                tr = rank_sq + dr_d * step
                on_b = ((tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8)).to(tl.int32)
                tsq = tl.where(on_b > 0, tr * 8 + tf, 0)
                tp = tl.gather(pieces_nk, tsq, axis=0)
                tp = tl.where(on_b > 0, tp, 0)
                is_piece = (tp != 0).to(tl.int32)
                first_p = walking * is_piece
                if is_orth:
                    matches = ((tp == rook_code) | (tp == queen_code)).to(tl.int32)
                else:
                    matches = ((tp == bishop_code) | (tp == queen_code)).to(tl.int32)
                slider_atk = slider_atk | (first_p * matches)
                walking = walking * (1 - is_piece) * on_b

        enemy_attacks = (knight_atk | king_atk | pawn_atk | slider_atk)  # [64] int32 0/1
        # Pack into a u64 bitboard (scalar). bit_per_sq used here and later.
        bit_per_sq = (tl.full([1], 1, tl.int64) << sq.to(tl.int64)).to(tl.int64)
        ea_bb = tl.sum(tl.where(enemy_attacks != 0, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))

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
        block_or_capture = tl.zeros([BLOCK_SQ], tl.int32)
        num_checkers_slider = 0

        # Pre-compute "is_orth direction" array as static (we'll inline via cond).
        for d_idx in tl.static_range(0, 8):
            df_d = DFS_C[d_idx]
            dr_d = DRS_C[d_idx]
            is_orth = ORTH_C[d_idx]

            # Walk 7 steps. Record first and second occupied square + values.
            first_step = -1
            second_step = -1
            first_sq = 0
            second_sq = 0
            first_p = 0
            second_p = 0
            # Need to track ray squares to build block_or_capture/pin_ray masks.
            # Store ray_sq and ray_piece per step in registers (7 each).
            rs_0 = 0; rs_1 = 0; rs_2 = 0; rs_3 = 0; rs_4 = 0; rs_5 = 0; rs_6 = 0
            on_0 = 0; on_1 = 0; on_2 = 0; on_3 = 0; on_4 = 0; on_5 = 0; on_6 = 0

            for step in tl.static_range(1, 8):
                tf = king_file + df_d * step
                tr = king_rank + dr_d * step
                on_b = ((tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8))
                tsq = tl.where(on_b, tr * 8 + tf, 0)
                # Use pieces (king kept) for blocker test from from-sq side, but for
                # king-ray analysis we use pieces (king kept too — but king is
                # the start). We use pieces (with king removed for the king's own
                # square only).
                # Load piece at tsq via gather over `pieces`.
                tp = tl.load(pieces_ptr + pid * BLOCK_SQ + tsq).to(tl.int32)
                # Cast on_b to scalar int by sum trick: it's already scalar here
                # because tf/tr are scalar (king_file/king_rank are scalars).
                tp = tl.where(on_b, tp, 0)

                is_piece = tp != 0
                # First-piece detection.
                is_first = is_piece & (first_step == -1)
                first_step = tl.where(is_first, step - 1, first_step)
                first_sq = tl.where(is_first, tsq, first_sq)
                first_p = tl.where(is_first, tp, first_p)
                # Second-piece detection (must come strictly after first).
                is_second = is_piece & (first_step != -1) & (step - 1 > first_step) & (second_step == -1)
                second_step = tl.where(is_second, step - 1, second_step)
                second_sq = tl.where(is_second, tsq, second_sq)
                second_p = tl.where(is_second, tp, second_p)

                # Stash ray sq + on_b.
                if step == 1:
                    rs_0 = tsq; on_0 = tl.where(on_b, 1, 0)
                if step == 2:
                    rs_1 = tsq; on_1 = tl.where(on_b, 1, 0)
                if step == 3:
                    rs_2 = tsq; on_2 = tl.where(on_b, 1, 0)
                if step == 4:
                    rs_3 = tsq; on_3 = tl.where(on_b, 1, 0)
                if step == 5:
                    rs_4 = tsq; on_4 = tl.where(on_b, 1, 0)
                if step == 6:
                    rs_5 = tsq; on_5 = tl.where(on_b, 1, 0)
                if step == 7:
                    rs_6 = tsq; on_6 = tl.where(on_b, 1, 0)

            # Classify first/second pieces.
            first_is_white = (first_p >= 1) & (first_p <= 6)
            first_is_black = (first_p >= 7) & (first_p <= 12)
            first_is_own = tl.where(is_white_mover, first_is_white, first_is_black)
            first_is_enemy = tl.where(is_white_mover, first_is_black, first_is_white)

            if is_orth:
                first_is_enemy_slider = first_is_enemy & ((first_p == rook_code) | (first_p == queen_code))
                second_is_enemy_slider = (
                    ((second_p >= 1) & (second_p <= 12))
                    & (tl.where(is_white_mover,
                                (second_p >= 7) & (second_p <= 12),
                                (second_p >= 1) & (second_p <= 6)))
                    & ((second_p == rook_code) | (second_p == queen_code))
                )
            else:
                first_is_enemy_slider = first_is_enemy & ((first_p == bishop_code) | (first_p == queen_code))
                second_is_enemy_slider = (
                    ((second_p >= 1) & (second_p <= 12))
                    & (tl.where(is_white_mover,
                                (second_p >= 7) & (second_p <= 12),
                                (second_p >= 1) & (second_p <= 6)))
                    & ((second_p == bishop_code) | (second_p == queen_code))
                )

            is_check_ray = first_is_enemy_slider & (first_step != -1)
            is_pin_ray = first_is_own & second_is_enemy_slider & (second_step != -1)
            num_checkers_slider = num_checkers_slider + tl.where(is_check_ray, 1, 0)

            # Build block_or_capture additions: squares 1..first_step+1 along ray
            # if check_ray. We have step indices 0..6; in_segment[k] = (k <= first_step).
            # Add (sq == ray_sq[k]) for k where in_segment.
            # When not check_ray, contribute nothing.
            for k in tl.static_range(0, 7):
                rsq_k = 0
                onb_k = 0
                if k == 0: rsq_k = rs_0; onb_k = on_0
                if k == 1: rsq_k = rs_1; onb_k = on_1
                if k == 2: rsq_k = rs_2; onb_k = on_2
                if k == 3: rsq_k = rs_3; onb_k = on_3
                if k == 4: rsq_k = rs_4; onb_k = on_4
                if k == 5: rsq_k = rs_5; onb_k = on_5
                if k == 6: rsq_k = rs_6; onb_k = on_6
                in_seg_check = (k <= first_step) & is_check_ray & (onb_k > 0)
                add_check = tl.where(in_seg_check & (sq == rsq_k), 1, 0)
                block_or_capture = block_or_capture | add_check

            # Pin direction (compact). For each direction d that is a pin ray,
            # write sign(df_d), sign(dr_d) into the row at first_sq.
            df_unit_d = (1 if df_d > 0 else (-1 if df_d < 0 else 0))
            dr_unit_d = (1 if dr_d > 0 else (-1 if dr_d < 0 else 0))
            pin_match_lane = (sq == first_sq) & is_pin_ray
            pin_df_per_from = tl.where(pin_match_lane, df_unit_d, pin_df_per_from)
            pin_dr_per_from = tl.where(pin_match_lane, dr_unit_d, pin_dr_per_from)

        # Knight + pawn checkers (single-square contributions to block_or_capture).
        # knight_attacks_king[sq] = KNIGHT_TBL[king_sq, sq] (symmetric).
        knight_row_idx = king_sq * 64 + sq
        knight_attacks_king = tl.load(KNIGHT_TBL_ptr + knight_row_idx).to(tl.int32)
        enemy_knight_at = (pieces == knight_code).to(tl.int32)
        knight_checkers = knight_attacks_king * enemy_knight_at

        # Pawn attackers TO king sq: use the source table — row king_sq of the
        # transposed table = WPAWN_ATTACK[:, king_sq] when enemy is white.
        # We don't have transposed table loaded; compute inline as TBL[sq, king_sq].
        # Mover white -> enemy black -> source pawns are black pawns -> use BPAWN_ATTACK[sq, king_sq].
        if True:  # always executed; conditional below
            pass
        pawn_src_idx = sq * 64 + king_sq
        bp_atk_to_king = tl.load(BPAWN_TBL_ptr + pawn_src_idx).to(tl.int32)
        wp_atk_to_king = tl.load(WPAWN_TBL_ptr + pawn_src_idx).to(tl.int32)
        pawn_atk_to_king = tl.where(is_white_mover, bp_atk_to_king, wp_atk_to_king)
        enemy_pawn_at = (pieces == pawn_code).to(tl.int32)
        pawn_checkers = pawn_atk_to_king * enemy_pawn_at

        block_or_capture = block_or_capture | knight_checkers | pawn_checkers
        boc_bb = tl.sum(tl.where(block_or_capture != 0, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))
        num_checkers = (
            num_checkers_slider
            + tl.sum(knight_checkers)
            + tl.sum(pawn_checkers)
        )
        in_check = num_checkers >= 1
        double_check = num_checkers >= 2

        # ----- Phase 4b: per-board scalar prep for castling and EP-pin -----
        # (Hoisted out of the per-(from_sq, plane) loop so they're computed once.)
        e1, f1c, g1c, d1c, c1c, b1c, a1c, h1c = 4, 5, 6, 3, 2, 1, 0, 7
        e8, f8c, g8c, d8c, c8c, b8c, a8c, h8c = 60, 61, 62, 59, 58, 57, 56, 63
        p_e1 = tl.sum(tl.where(sq == e1, pieces, 0))
        p_h1 = tl.sum(tl.where(sq == h1c, pieces, 0))
        p_a1 = tl.sum(tl.where(sq == a1c, pieces, 0))
        p_f1 = tl.sum(tl.where(sq == f1c, pieces, 0))
        p_g1 = tl.sum(tl.where(sq == g1c, pieces, 0))
        p_d1 = tl.sum(tl.where(sq == d1c, pieces, 0))
        p_c1 = tl.sum(tl.where(sq == c1c, pieces, 0))
        p_b1 = tl.sum(tl.where(sq == b1c, pieces, 0))
        p_e8 = tl.sum(tl.where(sq == e8, pieces, 0))
        p_h8 = tl.sum(tl.where(sq == h8c, pieces, 0))
        p_a8 = tl.sum(tl.where(sq == a8c, pieces, 0))
        p_f8 = tl.sum(tl.where(sq == f8c, pieces, 0))
        p_g8 = tl.sum(tl.where(sq == g8c, pieces, 0))
        p_d8 = tl.sum(tl.where(sq == d8c, pieces, 0))
        p_c8 = tl.sum(tl.where(sq == c8c, pieces, 0))
        p_b8 = tl.sum(tl.where(sq == b8c, pieces, 0))

        side_white = is_white_mover
        side_black = ~is_white_mover
        cr_wk = (cr & 1) != 0
        cr_wq = ((cr >> 1) & 1) != 0
        cr_bk = ((cr >> 2) & 1) != 0
        cr_bq = ((cr >> 3) & 1) != 0

        wk_ok = (
            side_white & cr_wk
            & (p_e1 == 6) & (p_h1 == 4) & (p_f1 == 0) & (p_g1 == 0)
            & (((ea_bb >> e1) & 1) == 0)
            & (((ea_bb >> f1c) & 1) == 0)
            & (((ea_bb >> g1c) & 1) == 0)
        )
        wq_ok = (
            side_white & cr_wq
            & (p_e1 == 6) & (p_a1 == 4) & (p_d1 == 0) & (p_c1 == 0) & (p_b1 == 0)
            & (((ea_bb >> e1) & 1) == 0)
            & (((ea_bb >> d1c) & 1) == 0)
            & (((ea_bb >> c1c) & 1) == 0)
        )
        bk_ok = (
            side_black & cr_bk
            & (p_e8 == 12) & (p_h8 == 10) & (p_f8 == 0) & (p_g8 == 0)
            & (((ea_bb >> e8) & 1) == 0)
            & (((ea_bb >> f8c) & 1) == 0)
            & (((ea_bb >> g8c) & 1) == 0)
        )
        bq_ok = (
            side_black & cr_bq
            & (p_e8 == 12) & (p_a8 == 10) & (p_d8 == 0) & (p_c8 == 0) & (p_b8 == 0)
            & (((ea_bb >> e8) & 1) == 0)
            & (((ea_bb >> d8c) & 1) == 0)
            & (((ea_bb >> c8c) & 1) == 0)
        )

        # ---- EP horizontal-pin: per-from ep_unsafe[64] (state-dependent) ----
        ep_valid = ep >= 0
        ep_cap_sq = tl.where(is_white_mover, ep - 8, ep + 8)
        ep_cap_rank = ep_cap_sq >> 3
        king_on_eprank = (king_rank == ep_cap_rank) & ep_valid
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

        # Occupancy bitboard (used inside chunk loop for slider blocker).
        occ_bb = tl.sum(tl.where(pieces != 0, bit_per_sq, tl.zeros([BLOCK_SQ], tl.int64)))

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

            # to-sq piece (gather from pieces).
            to_piece = tl.gather(
                pieces, real_to.reshape([BLOCK_SQ * BLOCK_PL_CHUNK]), axis=0
            ).reshape([BLOCK_SQ, BLOCK_PL_CHUNK])
            to_piece = tl.where(on_board2d, to_piece, 0)

            to_is_white = (to_piece >= 1) & (to_piece <= 6)
            to_is_black = (to_piece >= 7) & (to_piece <= 12)
            to_is_empty = to_piece == 0
            to_is_enemy = tl.where(is_white_mover, to_is_black, to_is_white)

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
        tbls["knight"], tbls["king"], tbls["wpawn"], tbls["bpawn"], tbls["between_bb"],
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
        num_warps=4,
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
