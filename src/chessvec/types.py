"""Shared constants and types for the chess engines.

Square indexing: 0..63, with 0 = a1, 7 = h1, 56 = a8, 63 = h8.
That is, ``square = rank * 8 + file`` with rank 0 = white's back rank.

Piece encoding (used in the reference 8x8 board): integer codes
0 = empty, 1..6 = white P,N,B,R,Q,K, 7..12 = black p,n,b,r,q,k.

Bitboard layout (used in the vectorized engine): plane index
0..5 = white P,N,B,R,Q,K; 6..11 = black p,n,b,r,q,k.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


class Piece(IntEnum):
    EMPTY = 0
    WP = 1
    WN = 2
    WB = 3
    WR = 4
    WQ = 5
    WK = 6
    BP = 7
    BN = 8
    BB = 9
    BR = 10
    BQ = 11
    BK = 12


PIECE_CHARS: Final[str] = ".PNBRQKpnbrqk"


def piece_char(p: int) -> str:
    return PIECE_CHARS[p]


def char_piece(c: str) -> int:
    return PIECE_CHARS.index(c)


def piece_color(p: int) -> int:
    """Return 0 for white, 1 for black, -1 for empty."""
    if p == 0:
        return -1
    return 0 if p <= 6 else 1


def piece_type(p: int) -> int:
    """Return 1..6 (P,N,B,R,Q,K) regardless of color, 0 for empty."""
    if p == 0:
        return 0
    return p if p <= 6 else p - 6


# ---------------------------------------------------------------------------
# Bitboard plane indices (used by vectorized engine)
# ---------------------------------------------------------------------------

# Plane indexing: white pawn=0, knight=1, bishop=2, rook=3, queen=4, king=5,
# black pawn=6, knight=7, bishop=8, rook=9, queen=10, king=11.
WP_PLANE: Final[int] = 0
WN_PLANE: Final[int] = 1
WB_PLANE: Final[int] = 2
WR_PLANE: Final[int] = 3
WQ_PLANE: Final[int] = 4
WK_PLANE: Final[int] = 5
BP_PLANE: Final[int] = 6
BN_PLANE: Final[int] = 7
BB_PLANE: Final[int] = 8
BR_PLANE: Final[int] = 9
BQ_PLANE: Final[int] = 10
BK_PLANE: Final[int] = 11

NUM_PLANES: Final[int] = 12

WHITE: Final[int] = 0
BLACK: Final[int] = 1


# ---------------------------------------------------------------------------
# Castling-rights bit indices: bit per (color, side).
# 0 = white kingside (h1), 1 = white queenside (a1),
# 2 = black kingside (h8), 3 = black queenside (a8).
# ---------------------------------------------------------------------------

CR_WK: Final[int] = 0
CR_WQ: Final[int] = 1
CR_BK: Final[int] = 2
CR_BQ: Final[int] = 3


# ---------------------------------------------------------------------------
# Game-result codes (returned by both engines).
# ---------------------------------------------------------------------------

ONGOING: Final[int] = 0
WHITE_WIN: Final[int] = 1
BLACK_WIN: Final[int] = 2
DRAW: Final[int] = 3


# ---------------------------------------------------------------------------
# Action space.
#
# We use a 4672-dim AlphaZero-style action space: 64 from-squares times 73
# move planes. The 73 planes are:
#   0..55  : "queen-like" moves -- 8 directions * 7 distances. Direction order
#            is (N, NE, E, SE, S, SW, W, NW); index = dir * 7 + (distance-1).
#   56..63 : 8 knight moves, in a fixed order.
#   64..72 : 9 underpromotions = 3 capture-files (left, straight, right) *
#            3 promotion pieces (knight, bishop, rook). Queen promotions use
#            the queen-like planes.
# ---------------------------------------------------------------------------

NUM_FROM_SQUARES: Final[int] = 64
NUM_MOVE_PLANES: Final[int] = 73
ACTION_SIZE: Final[int] = NUM_FROM_SQUARES * NUM_MOVE_PLANES  # 4672

# Direction deltas (df, dr) for the 8 queen-like directions.
QUEEN_DIRS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),  # N
    (1, 1),  # NE
    (1, 0),  # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1),  # SW
    (-1, 0),  # W
    (-1, 1),  # NW
)

# Knight deltas in a fixed order (df, dr).
KNIGHT_DIRS: Final[tuple[tuple[int, int], ...]] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# Underpromotion: capture-file deltas (left, straight, right) from white's POV.
UNDERPROMO_FILES: Final[tuple[int, ...]] = (-1, 0, 1)
# Promotion piece ordering: 0=knight, 1=bishop, 2=rook.
UNDERPROMO_PIECES: Final[tuple[int, ...]] = (2, 3, 4)  # piece-type ids: N,B,R
