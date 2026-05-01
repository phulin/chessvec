"""Conversion between (from_sq, to_sq, promotion) moves and the 4672-dim
AlphaZero-style action index used by both engines.

For black's POV the move planes are mirrored vertically: a black move with
delta (df, dr) lives in the same plane as a white move with delta (df, -dr).
This keeps the "to-move player always advances forward" convention used by
self-play frameworks. Both engines must use the same convention; the helpers
here handle it.
"""

from __future__ import annotations

from .types import (
    ACTION_SIZE,
    BLACK,
    KNIGHT_DIRS,
    NUM_MOVE_PLANES,
    QUEEN_DIRS,
    UNDERPROMO_FILES,
    UNDERPROMO_PIECES,
    WHITE,
)


def square(file: int, rank: int) -> int:
    return rank * 8 + file


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def encode_move(
    from_sq: int,
    to_sq: int,
    promotion: int,  # 0 if none, else piece-type id 2..5 (N,B,R,Q)
    side_to_move: int,
) -> int:
    """Encode a (from, to, promo) tuple to an action index in [0, 4672).

    Raises ValueError if the move does not fit any plane.
    """
    f0, r0 = file_of(from_sq), rank_of(from_sq)
    f1, r1 = file_of(to_sq), rank_of(to_sq)
    df, dr = f1 - f0, r1 - r0

    # Mirror dr if we're encoding from black's POV.
    if side_to_move == BLACK:
        dr = -dr

    # Underpromotion (knight/bishop/rook): use planes 64..72.
    if promotion in (2, 3, 4):  # N,B,R
        # Underpromotion only happens on a 1-square forward push or capture.
        if df not in UNDERPROMO_FILES:
            raise ValueError(f"bad underpromotion df={df}")
        file_idx = UNDERPROMO_FILES.index(df)
        piece_idx = UNDERPROMO_PIECES.index(promotion)
        plane = 64 + piece_idx * 3 + file_idx
        return from_sq * NUM_MOVE_PLANES + plane

    # Queen promotion or non-promotion queen-like move.
    # Try the 8 queen directions:
    for d_idx, (qdf, qdr) in enumerate(QUEEN_DIRS):
        if qdf == 0 and qdr == 0:
            continue
        if df * qdr == dr * qdf and (df * qdf >= 0 and dr * qdr >= 0):
            # Same direction. Distance along ray:
            dist = max(abs(df), abs(dr))
            if 1 <= dist <= 7 and df == qdf * dist and dr == qdr * dist:
                plane = d_idx * 7 + (dist - 1)
                return from_sq * NUM_MOVE_PLANES + plane

    # Knight planes 56..63.
    for k_idx, (kdf, kdr) in enumerate(KNIGHT_DIRS):
        if df == kdf and dr == kdr:
            plane = 56 + k_idx
            return from_sq * NUM_MOVE_PLANES + plane

    raise ValueError(f"move from {from_sq} to {to_sq} (promo={promotion}) not encodable")


def decode_move(action: int, side_to_move: int) -> tuple[int, int, int]:
    """Decode an action index into (from_sq, to_sq, promotion).

    Promotion is 0 if no promotion or queen promotion (caller resolves whether
    the move is a pawn reaching the last rank, in which case it is a queen
    promotion). Returns 2/3/4 for explicit underpromotions to N/B/R.
    """
    if not 0 <= action < ACTION_SIZE:
        raise ValueError(f"action {action} out of range")
    from_sq = action // NUM_MOVE_PLANES
    plane = action % NUM_MOVE_PLANES
    f0, r0 = file_of(from_sq), rank_of(from_sq)

    if plane < 56:  # queen-like
        d_idx, dist_m1 = divmod(plane, 7)
        dist = dist_m1 + 1
        df, dr = QUEEN_DIRS[d_idx]
        df *= dist
        dr *= dist
        promotion = 0
    elif plane < 64:  # knight
        df, dr = KNIGHT_DIRS[plane - 56]
        promotion = 0
    else:  # underpromotion
        u = plane - 64
        piece_idx, file_idx = divmod(u, 3)
        df = UNDERPROMO_FILES[file_idx]
        dr = 1  # promotion is always a one-square advance (in mover's frame)
        promotion = UNDERPROMO_PIECES[piece_idx]

    # Un-mirror for black.
    if side_to_move == BLACK:
        dr = -dr

    f1, r1 = f0 + df, r0 + dr
    if not (0 <= f1 < 8 and 0 <= r1 < 8):
        # Off-board planes for edge squares are simply unreachable; we still
        # return a sentinel so callers can mask them.
        return from_sq, -1, promotion
    to_sq = square(f1, r1)
    return from_sq, to_sq, promotion


def is_promotion_rank(side: int, to_sq: int) -> bool:
    r = rank_of(to_sq)
    return (side == WHITE and r == 7) or (side == BLACK and r == 0)
