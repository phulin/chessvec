"""Tests for the vectorized engine in isolation (smoke + sanity checks)."""

from __future__ import annotations

import torch

from chessvec.reference import State
from chessvec.types import BLACK_WIN, DRAW, ONGOING
from chessvec.vectorized import (
    apply_action,
    from_states,
    game_result,
    legal_action_mask,
    starting_state,
    to_planes,
    to_states,
)


def test_starting_state_has_20_legal_moves() -> None:
    vs = starting_state(4)
    legal = legal_action_mask(vs)
    counts = legal.sum(dim=1)
    assert torch.all(counts == 20), f"got {counts.tolist()}"


def test_state_round_trip_via_states() -> None:
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/pppqpppp/2nb1n2/3p4/3P4/2NB1N2/PPPQPPPP/R3K2R w KQkq - 0 1",
    ]
    states = [State.from_fen(f) for f in fens]
    vs = from_states(states)
    back = to_states(vs)
    for original, recovered in zip(states, back, strict=True):
        assert original.to_fen() == recovered.to_fen()


def test_to_planes_matches_pieces() -> None:
    vs = starting_state(2)
    planes = to_planes(vs.pieces)
    assert planes.shape == (2, 12, 64)
    # Plane 0 (white pawns) should have 8 pawns on rank 2 (squares 8..15).
    assert planes[0, 0].sum() == 8
    assert planes[0, 0, 8:16].all()
    # Plane 5 (white king) should have 1 piece on e1 (sq 4).
    assert planes[0, 5].sum() == 1
    assert planes[0, 5, 4]


def test_apply_action_starting_e2e4() -> None:
    from chessvec.action_encoding import encode_move
    from chessvec.types import WHITE

    vs = starting_state(1)
    e2 = 12
    e4 = 28
    action = encode_move(e2, e4, 0, WHITE)
    new = apply_action(vs, torch.tensor([action], dtype=torch.long))
    after = to_states(new)[0]
    assert after.board[12] == 0
    assert after.board[28] == 1  # white pawn
    assert after.side_to_move == 1
    assert after.en_passant == 20  # e3


def test_game_result_all_ongoing_at_start() -> None:
    vs = starting_state(8)
    assert torch.all(game_result(vs) == ONGOING)


def test_game_result_checkmate() -> None:
    fool = State.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    vs = from_states([fool])
    assert game_result(vs).item() == BLACK_WIN


def test_game_result_stalemate() -> None:
    sm = State.from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    vs = from_states([sm])
    assert game_result(vs).item() == DRAW
