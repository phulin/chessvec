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
