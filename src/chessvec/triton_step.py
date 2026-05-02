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
        BLOCK_SQ: tl.constexpr,    # 64
        BLOCK_PL: tl.constexpr,    # 128 (PAD)
        NUM_PL: tl.constexpr,      # 73
    ):
        pid = tl.program_id(axis=0)
        sq = tl.arange(0, BLOCK_SQ)              # [64]
        pl = tl.arange(0, BLOCK_PL)              # [128]
        plane_valid = pl < NUM_PL                # [128]

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
        num_checkers = (
            num_checkers_slider
            + tl.sum(knight_checkers)
            + tl.sum(pawn_checkers)
        )
        in_check = num_checkers >= 1
        double_check = num_checkers >= 2

        # ----- Phase 5: per-(from_sq, plane) pseudo-legal & legality -----
        # Load plane-geom tables [64, BLOCK_PL].
        rows2d = tl.arange(0, BLOCK_SQ)[:, None]
        cols2d = tl.arange(0, BLOCK_PL)[None, :]
        flat_idx = rows2d * BLOCK_PL + cols2d  # [64, PAD]

        df = tl.load(df_tbl_ptr + flat_idx).to(tl.int32)
        dr = tl.load(dr_tbl_ptr + flat_idx).to(tl.int32)
        kind_v = tl.load(kind_tbl_ptr + flat_idx).to(tl.int32)
        ray_dir = tl.load(ray_dir_tbl_ptr + flat_idx).to(tl.int32)
        ray_dist = tl.load(ray_dist_tbl_ptr + flat_idx).to(tl.int32)
        promo = tl.load(promo_tbl_ptr + flat_idx).to(tl.int32)

        # Mover-frame -> board-frame dr.
        dr_signed = tl.where(is_white_mover, dr, -dr)
        # Real from-square per row, real-to per (row, plane).
        rfile_from = (rows2d & 7).to(tl.int32)
        rrank_from = (rows2d >> 3).to(tl.int32)
        rfile_to = rfile_from + df
        rrank_to = rrank_from + dr_signed
        on_board2d = (rfile_to >= 0) & (rfile_to < 8) & (rrank_to >= 0) & (rrank_to < 8)
        real_to = tl.where(on_board2d, rrank_to * 8 + rfile_to, 0)  # int32 [64, PAD]

        # Piece at from-sq (per row) broadcast over cols.
        from_piece = pieces[:, None] + tl.zeros([BLOCK_SQ, BLOCK_PL], tl.int32)

        to_piece_flat = tl.gather(pieces, real_to.reshape([BLOCK_SQ * BLOCK_PL]), axis=0)
        to_piece = to_piece_flat.reshape([BLOCK_SQ, BLOCK_PL])
        to_piece = tl.where(on_board2d, to_piece, 0)

        # Mover/enemy classification at from and to.
        from_is_white = (from_piece >= 1) & (from_piece <= 6)
        from_is_black = (from_piece >= 7) & (from_piece <= 12)
        from_belongs_mover = tl.where(is_white_mover, from_is_white, from_is_black)

        to_is_white = (to_piece >= 1) & (to_piece <= 6)
        to_is_black = (to_piece >= 7) & (to_piece <= 12)
        to_is_empty = to_piece == 0
        to_is_enemy = tl.where(is_white_mover, to_is_black, to_is_white)

        pt = tl.where(from_is_black, from_piece - 6,
                       tl.where(from_is_white, from_piece, 0))  # 0..6

        # ---- Plane-allowed per piece type ----
        # Queen-like (kind==0):
        #   pawn (pt=1): allowed if (dir==0 & dist∈{1,2}) | (dir∈{1,7} & dist==1)
        #   knight: not allowed (knight uses kind==1)
        #   bishop (pt=3): dir is diag (1,3,5,7)
        #   rook (pt=4): dir is orth (0,2,4,6)
        #   queen (pt=5): any
        #   king (pt=6): any with dist==1 (castling: dist==2 too — handled via override later)
        # Knight (kind==1): only piece type knight (pt=2).
        # Underpromo (kind==2): pawn (pt=1).
        is_q = kind_v == 0
        is_k = kind_v == 1
        is_u = kind_v == 2

        # Allowed-by-piece base: (will refine below)
        is_diag_dir = (ray_dir == 1) | (ray_dir == 3) | (ray_dir == 5) | (ray_dir == 7)
        is_orth_dir = (ray_dir == 0) | (ray_dir == 2) | (ray_dir == 4) | (ray_dir == 6)

        is_pawn = pt == 1
        is_knight = pt == 2
        is_bishop = pt == 3
        is_rook = pt == 4
        is_queen = pt == 5
        is_king_p = pt == 6

        pawn_q_ok = is_q & (
            ((ray_dir == 0) & ((ray_dist == 1) | (ray_dist == 2)))
            | (((ray_dir == 1) | (ray_dir == 7)) & (ray_dist == 1))
        )
        bishop_q_ok = is_q & is_diag_dir
        rook_q_ok = is_q & is_orth_dir
        queen_q_ok = is_q
        king_q_ok = is_q & (ray_dist == 1)
        knight_k_ok = is_k

        plane_allowed = (
            (is_pawn & (pawn_q_ok | is_u))
            | (is_knight & knight_k_ok)
            | (is_bishop & bishop_q_ok)
            | (is_rook & rook_q_ok)
            | (is_queen & queen_q_ok)
            | (is_king_p & king_q_ok)
        )

        # ---- Slider intermediate-blocker test ----
        # For queen-like planes with dist >= 2: any of squares 1..dist-1 along
        # board-frame direction occupied?
        df_unit = tl.where(df > 0, 1, tl.where(df < 0, -1, 0))
        dr_unit_mover = tl.where(dr > 0, 1, tl.where(dr < 0, -1, 0))
        dr_unit = tl.where(is_white_mover, dr_unit_mover, -dr_unit_mover)

        has_intermediate = tl.zeros([BLOCK_SQ, BLOCK_PL], tl.int32)
        for step in tl.static_range(1, 7):  # 1..6 covers dist-1 up to 6
            # in segment iff step <= ray_dist - 1 (i.e., strictly before to_sq).
            in_seg = (is_q & (ray_dist >= 2) & (step <= ray_dist - 1))
            tf2 = rfile_from + df_unit * step
            tr2 = rrank_from + dr_unit * step
            on_b2 = (tf2 >= 0) & (tf2 < 8) & (tr2 >= 0) & (tr2 < 8)
            tsq2 = tl.where(on_b2, tr2 * 8 + tf2, 0)
            tp2 = tl.gather(pieces, tsq2.reshape([BLOCK_SQ * BLOCK_PL]), axis=0).reshape([BLOCK_SQ, BLOCK_PL])
            tp2 = tl.where(on_b2, tp2, 0)
            occ_step = (tp2 != 0).to(tl.int32) * tl.where(in_seg, 1, 0)
            has_intermediate = has_intermediate | occ_step

        # ---- Compose pseudo-legal base ----
        base = (
            from_belongs_mover & plane_allowed & on_board2d
            & (to_is_empty | to_is_enemy)
        )
        base = base & ~(is_q & (has_intermediate > 0))

        # ---- Pawn-specific filtering ----
        is_pawn_push1 = is_q & (ray_dir == 0) & (ray_dist == 1)
        is_pawn_push2 = is_q & (ray_dir == 0) & (ray_dist == 2)
        is_pawn_cap_e = is_q & (ray_dir == 1) & (ray_dist == 1)
        is_pawn_cap_w = is_q & (ray_dir == 7) & (ray_dist == 1)
        underpromo_push = is_u & (df == 0)
        underpromo_cap = is_u & (df != 0)
        push_planes = is_pawn_push1 | is_pawn_push2 | underpromo_push
        cap_planes = is_pawn_cap_e | is_pawn_cap_w | underpromo_cap

        # Forbid pawn pushes onto non-empty squares.
        base = base & ~(is_pawn & push_planes & ~to_is_empty)

        # Forbid pawn captures onto non-enemy unless it's en passant.
        is_ep_target_cell = (ep >= 0) & (real_to == ep)
        base = base & ~(is_pawn & cap_planes & ~(to_is_enemy | is_ep_target_cell))

        # Pawn double-push: from rank must be mover-frame rank 1; intermediate empty.
        from_rank_mover = tl.where(is_white_mover, rrank_from, 7 - rrank_from)
        bad_push2_rank = is_pawn & is_pawn_push2 & (from_rank_mover != 1)
        base = base & ~bad_push2_rank
        # Intermediate-square-empty for double push: from_sq + 1 forward in mover frame.
        forward = tl.where(is_white_mover, 8, -8)
        inter_sq_1d = tl.minimum(tl.maximum(sq + forward, 0), 63)  # [64]
        inter_piece_1d = tl.load(pieces_ptr + pid * BLOCK_SQ + inter_sq_1d).to(tl.int32)
        inter_piece = inter_piece_1d[:, None] + tl.zeros([BLOCK_SQ, BLOCK_PL], tl.int32)
        bad_push2_blocked = is_pawn & is_pawn_push2 & (inter_piece != 0)
        base = base & ~bad_push2_blocked

        # Underpromo destination must be mover-frame rank 7.
        to_rank_mover = tl.where(is_white_mover, rrank_to, 7 - rrank_to)
        bad_underpromo = is_pawn & is_u & (to_rank_mover != 7) & on_board2d
        base = base & ~bad_underpromo

        # ---- King castling: disable naive king dist-2 moves; re-enable legal ones below.
        is_castle_plane = is_q & ((ray_dir == 2) | (ray_dir == 6)) & (ray_dist == 2)
        base = base & ~(is_king_p & is_castle_plane)

        # ---- Legality filter ----
        # King moves: to_sq must not be in enemy_attacks (king-removed).
        # Cast enemy_attacks to a [64, PAD] mask via gather over real_to.
        ea_at_to = tl.gather(enemy_attacks, real_to.reshape([BLOCK_SQ * BLOCK_PL]), axis=0).reshape([BLOCK_SQ, BLOCK_PL])
        from_is_king_2d = (from_piece == own_king_code)
        king_legal = ea_at_to == 0

        # Non-king: forbidden in double check; if in single check, must land on
        # block_or_capture; respect pin.
        boc_at_to = tl.gather(block_or_capture, real_to.reshape([BLOCK_SQ * BLOCK_PL]), axis=0).reshape([BLOCK_SQ, BLOCK_PL])

        # Pin legality (compact). Per (from_sq, plane): if the from-sq is pinned
        # along a direction d (board frame), the move is legal iff its unit
        # direction equals ±d. Knight moves can never satisfy a single ray
        # direction, so a pinned knight is always rejected.
        pin_df_b = pin_df_per_from[:, None]  # [64, 1]
        pin_dr_b = pin_dr_per_from[:, None]
        pinned = pin_df_b != 99
        # Move unit-vector in board frame.
        m_df_u = tl.where(df > 0, 1, tl.where(df < 0, -1, 0))
        m_dr_u_mover = tl.where(dr > 0, 1, tl.where(dr < 0, -1, 0))
        m_dr_u_board = tl.where(is_white_mover, m_dr_u_mover, -m_dr_u_mover)
        same = (m_df_u == pin_df_b) & (m_dr_u_board == pin_dr_b)
        opp = (m_df_u == -pin_df_b) & (m_dr_u_board == -pin_dr_b)
        pin_at_ft = ~pinned | ((same | opp) & ~is_k)

        non_king_legal = (
            (~double_check)
            & pin_at_ft
            & (~in_check | (boc_at_to != 0))
        )

        legal_flag = tl.where(from_is_king_2d, king_legal, non_king_legal)
        out_mask = base & legal_flag & plane_valid[None, :]

        # ---- Phase 6: castling (re-enable specific 4 castle planes) ----
        # Determine castle legality scalars.
        # Identify squares.
        e1, f1c, g1c, d1c, c1c, b1c, a1c, h1c = 4, 5, 6, 3, 2, 1, 0, 7
        e8, f8c, g8c, d8c, c8c, b8c, a8c, h8c = 60, 61, 62, 59, 58, 57, 56, 63
        # Pieces at specific squares (scalar via masked sum).
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

        ea_at_e1 = tl.sum(tl.where(sq == e1, enemy_attacks, 0))
        ea_at_f1 = tl.sum(tl.where(sq == f1c, enemy_attacks, 0))
        ea_at_g1 = tl.sum(tl.where(sq == g1c, enemy_attacks, 0))
        ea_at_d1 = tl.sum(tl.where(sq == d1c, enemy_attacks, 0))
        ea_at_c1 = tl.sum(tl.where(sq == c1c, enemy_attacks, 0))
        ea_at_e8 = tl.sum(tl.where(sq == e8, enemy_attacks, 0))
        ea_at_f8 = tl.sum(tl.where(sq == f8c, enemy_attacks, 0))
        ea_at_g8 = tl.sum(tl.where(sq == g8c, enemy_attacks, 0))
        ea_at_d8 = tl.sum(tl.where(sq == d8c, enemy_attacks, 0))
        ea_at_c8 = tl.sum(tl.where(sq == c8c, enemy_attacks, 0))

        wk_ok = (
            side_white & cr_wk
            & (p_e1 == 6) & (p_h1 == 4)
            & (p_f1 == 0) & (p_g1 == 0)
            & (ea_at_e1 == 0) & (ea_at_f1 == 0) & (ea_at_g1 == 0)
        )
        wq_ok = (
            side_white & cr_wq
            & (p_e1 == 6) & (p_a1 == 4)
            & (p_d1 == 0) & (p_c1 == 0) & (p_b1 == 0)
            & (ea_at_e1 == 0) & (ea_at_d1 == 0) & (ea_at_c1 == 0)
        )
        bk_ok = (
            side_black & cr_bk
            & (p_e8 == 12) & (p_h8 == 10)
            & (p_f8 == 0) & (p_g8 == 0)
            & (ea_at_e8 == 0) & (ea_at_f8 == 0) & (ea_at_g8 == 0)
        )
        bq_ok = (
            side_black & cr_bq
            & (p_e8 == 12) & (p_a8 == 10)
            & (p_d8 == 0) & (p_c8 == 0) & (p_b8 == 0)
            & (ea_at_e8 == 0) & (ea_at_d8 == 0) & (ea_at_c8 == 0)
        )

        # Plane indices for castle E (dir=2, dist=2 -> 2*7+1 = 15) and W (43).
        PLANE_CE = 2 * 7 + 1
        PLANE_CW = 6 * 7 + 1
        is_ce = (rows2d == 4) & (cols2d == PLANE_CE) & side_white  # e1
        is_cw = (rows2d == 4) & (cols2d == PLANE_CW) & side_white
        is_ce_b = (rows2d == 60) & (cols2d == PLANE_CE) & side_black  # e8
        is_cw_b = (rows2d == 60) & (cols2d == PLANE_CW) & side_black
        out_mask = out_mask | (is_ce & wk_ok)
        out_mask = out_mask | (is_cw & wq_ok)
        out_mask = out_mask | (is_ce_b & bk_ok)
        out_mask = out_mask | (is_cw_b & bq_ok)

        # ---- Phase 7: en-passant horizontal pin ----
        # If ep >= 0 and king is on the rank of ep_cap_sq: for each from_sq that
        # is making an ep capture, check if removing both vacated pawns exposes
        # the king to a horizontal R/Q.
        ep_valid = ep >= 0
        ep_cap_sq = tl.where(is_white_mover, ep - 8, ep + 8)
        ep_cap_rank = ep_cap_sq >> 3
        # is_ep_move per (from_sq, plane): mover pawn, cap plane, real_to == ep.
        mover_pawn_code = tl.where(is_white_mover, 1, 7)
        is_ep_move = (
            (out_mask != 0)
            & (real_to == ep) & (ep >= 0) & on_board2d
            & (from_piece == mover_pawn_code)
        )

        # Build per-from ep_unsafe[64].
        # Run only when there's any ep move possible AND king is on ep_cap_rank.
        king_on_eprank = (king_rank == ep_cap_rank) & ep_valid
        enemy_rq_v = ((pieces == rook_code) | (pieces == queen_code)).to(tl.int32)
        # For each from_sq (sq lane), compute west_first_sq and east_first_sq
        # on king's rank, skipping squares == from_sq, == ep_cap_sq, == king_sq.
        # Use the [64, 64] from-vs-target structure.
        from_lane = tl.arange(0, BLOCK_SQ)[:, None]   # from_sq
        sq_lane = tl.arange(0, BLOCK_SQ)[None, :]     # candidate square
        rank_of_sq = (sq_lane >> 3).to(tl.int32)
        file_of_sq = (sq_lane & 7).to(tl.int32)
        on_king_rank2 = rank_of_sq == king_rank
        vacated2 = (sq_lane == from_lane) | (sq_lane == ep_cap_sq) | (sq_lane == king_sq)
        occ2 = ((pieces[None, :] != 0) & on_king_rank2 & ~vacated2).to(tl.int32)
        west_occ = occ2 & (file_of_sq < king_file).to(tl.int32)
        east_occ = occ2 & (file_of_sq > king_file).to(tl.int32)
        # max file among west_occ, sentinel -1 -> use -1 + 1 trick by clamping below.
        west_file_v = tl.where(west_occ > 0, file_of_sq, -1)
        west_first_file = tl.max(west_file_v, axis=1)            # [64]
        east_file_v = tl.where(east_occ > 0, file_of_sq, 8)
        east_first_file = tl.min(east_file_v, axis=1)            # [64]
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
        ep_unsafe_2d = ep_unsafe_per_from[:, None] + tl.zeros([BLOCK_SQ, BLOCK_PL], tl.int32)

        out_mask = out_mask & ~(is_ep_move & (ep_unsafe_2d != 0))

        # ---- Write output ----
        # Output shape: [B, 64, BLOCK_PL] int8. We'll slice to 73 in Python.
        out_offset = pid * BLOCK_SQ * BLOCK_PL + rows2d * BLOCK_PL + cols2d
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
    _legal_mask_kernel[(B,)](
        pieces, side, cr, ep, out,
        tbls["df"], tbls["dr"], tbls["kind"], tbls["ray_dir"], tbls["ray_dist"], tbls["promo"],
        tbls["knight"], tbls["king"], tbls["wpawn"], tbls["bpawn"],
        BLOCK_SQ=64, BLOCK_PL=_PADDED_PLANES, NUM_PL=NUM_MOVE_PLANES,
        num_warps=16,
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
