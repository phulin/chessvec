"""Parity tests: Triton single-kernel step vs the PyTorch apply_action."""

from __future__ import annotations

import pytest
import torch

from chessvec.reference import State
from chessvec.vectorized import apply_action, from_states, legal_action_mask

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("Triton step requires CUDA", allow_module_level=True)

from chessvec.triton_step import triton_step  # noqa: E402


# A spread of positions exercising en passant, castling rights, captures,
# promotion, and double-push.
_FENS = [
    # Starting position.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Black to move, en passant available on e3.
    "rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 2",
    # Castling-rights-rich middlegame.
    "r3k2r/pppqpppp/2nb1n2/3p4/3P4/2NB1N2/PPPQPPPP/R3K2R w KQkq - 0 1",
    # White pawn one move from promotion.
    "8/P7/8/8/8/8/8/4k2K w - - 0 1",
    # Black pawn one move from promotion (underpromo capture available).
    "4k3/8/8/8/8/8/1p6/RN1K4 b - - 0 1",
    # Endgame with checks and limited mobility.
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def _states_to_cuda(fens: list[str]):
    states = [State.from_fen(f) for f in fens]
    return from_states(states, device="cuda")


def _assert_states_equal(a, b) -> None:
    assert torch.equal(a.pieces, b.pieces), (a.pieces, b.pieces)
    assert torch.equal(a.side_to_move, b.side_to_move)
    assert torch.equal(a.castling, b.castling)
    assert torch.equal(a.en_passant, b.en_passant)
    assert torch.equal(a.halfmove_clock, b.halfmove_clock)
    assert torch.equal(a.fullmove_number, b.fullmove_number)


def _first_legal_actions(vs) -> torch.Tensor:
    legal = legal_action_mask(vs)
    # Pick the first legal action per env. argmax on a bool tensor returns the
    # index of the first True (argmax stops at the first max value).
    return legal.long().argmax(dim=1)


def test_triton_step_matches_apply_action_on_first_legal_move() -> None:
    vs = _states_to_cuda(_FENS)
    actions = _first_legal_actions(vs)
    ref = apply_action(vs, actions)
    got = triton_step(vs, actions)
    _assert_states_equal(ref, got)


def test_triton_step_matches_over_random_rollout() -> None:
    """Multi-step rollout: alternate engines on every move and verify they
    agree at every step.
    """
    torch.manual_seed(0)
    vs = _states_to_cuda(_FENS)
    for _ in range(12):
        legal = legal_action_mask(vs)
        if not legal.any(dim=1).all():
            break
        # Pick a pseudo-random legal action per env.
        weights = legal.float() * torch.rand_like(legal.float())
        actions = weights.argmax(dim=1)
        ref = apply_action(vs, actions)
        got = triton_step(vs, actions)
        _assert_states_equal(ref, got)
        vs = ref
