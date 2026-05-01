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

from .types import NUM_MOVE_PLANES
from .vectorized import VState, _PLANE_GEOM

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


_TABLES_CACHE: dict[torch.device, tuple[Tensor, Tensor, Tensor, Tensor]] = {}


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
