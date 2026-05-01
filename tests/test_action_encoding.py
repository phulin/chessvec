"""Round-trip tests for the AlphaZero action encoding."""

from __future__ import annotations

import pytest

from chessvec.action_encoding import decode_move, encode_move
from chessvec.types import ACTION_SIZE, BLACK, WHITE


# Build a list of (from, to, promo, side) triples we care about.
def _gen_cases() -> list[tuple[int, int, int, int]]:
    cases: list[tuple[int, int, int, int]] = []
    # Queen-like moves from every square in every direction at distance 1 and 7.
    for fr in range(64):
        for to in range(64):
            if fr == to:
                continue
            df = (to & 7) - (fr & 7)
            dr = (to >> 3) - (fr >> 3)
            if df == 0 or dr == 0 or abs(df) == abs(dr):
                # queen-like
                for side in (WHITE, BLACK):
                    cases.append((fr, to, 0, side))
    # Underpromotions on rank 7 / rank 0 for white / black.
    for fr_file in range(8):
        for df in (-1, 0, 1):
            to_file = fr_file + df
            if not 0 <= to_file < 8:
                continue
            for promo in (2, 3, 4):
                cases.append((6 * 8 + fr_file, 7 * 8 + to_file, promo, WHITE))
                cases.append((1 * 8 + fr_file, 0 * 8 + to_file, promo, BLACK))
    return cases


@pytest.mark.parametrize("fr,to,promo,side", _gen_cases()[:200])
def test_encode_decode_round_trip(fr: int, to: int, promo: int, side: int) -> None:
    a = encode_move(fr, to, promo, side)
    assert 0 <= a < ACTION_SIZE
    fr2, to2, promo2 = decode_move(a, side)
    assert fr2 == fr
    assert to2 == to
    if promo in (2, 3, 4):
        assert promo2 == promo
    else:
        assert promo2 == 0
