"""Parity tests: the vectorized engine and the reference engine should
return identical results for legal-action mask, post-move state, and game
result on identical inputs."""

from __future__ import annotations

import random

import pytest
import torch

from chessvec import reference as ref
from chessvec import vectorized as vec
from chessvec.action_encoding import encode_move

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",  # checkmate
    "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1",  # stalemate
]


def _ref_mask(state: ref.State) -> torch.Tensor:
    return torch.tensor(ref.legal_action_mask(state), dtype=torch.bool)


@pytest.mark.parametrize("fen", FENS)
def test_legal_mask_parity(fen: str) -> None:
    state = ref.State.from_fen(fen)
    vs = vec.from_states([state])
    vec_mask = vec.legal_action_mask(vs)[0].cpu()
    ref_mask = _ref_mask(state)
    diff = (vec_mask ^ ref_mask).nonzero().flatten().tolist()
    assert vec_mask.equal(ref_mask), f"mismatch at {fen}; differing actions: {diff[:10]}"


@pytest.mark.parametrize("fen", FENS)
def test_apply_action_parity(fen: str) -> None:
    state = ref.State.from_fen(fen)
    moves = ref.legal_moves(state)
    if not moves:
        return
    vs = vec.from_states([state] * len(moves))
    actions: list[int] = []
    for mv in moves:
        promo_arg = 0 if mv[2] in (0, 5) else mv[2]
        actions.append(encode_move(mv[0], mv[1], promo_arg, state.side_to_move))
    new_vs = vec.apply_action(vs, torch.tensor(actions, dtype=torch.long))
    new_states = vec.to_states(new_vs)
    for mv, vec_state in zip(moves, new_states, strict=True):
        ref_state = ref.apply_move(state, mv)
        assert ref_state.to_fen() == vec_state.to_fen(), (
            f"diff after move {mv} from {fen}\n"
            f"  ref: {ref_state.to_fen()}\n  vec: {vec_state.to_fen()}"
        )


@pytest.mark.parametrize("fen", FENS)
def test_game_result_parity(fen: str) -> None:
    state = ref.State.from_fen(fen)
    vs = vec.from_states([state])
    assert vec.game_result(vs).item() == ref.game_result(state)


def test_random_walk_parity() -> None:
    """Play a few random games in lockstep through both engines, asserting
    legal masks and resulting states match at every ply."""
    rng = random.Random(0)
    n_games = 4
    states = [ref.State.starting() for _ in range(n_games)]
    for _ply in range(40):
        # Stop any games that have ended.
        active = [i for i, s in enumerate(states) if ref.game_result(s) == 0]
        if not active:
            break
        # Vectorized batch over active.
        vs = vec.from_states([states[i] for i in active])
        vec_mask = vec.legal_action_mask(vs).cpu()
        for k, i in enumerate(active):
            ref_mask = _ref_mask(states[i])
            assert vec_mask[k].equal(ref_mask), (
                f"mask diff at ply {_ply}, game {i}\n  fen: {states[i].to_fen()}"
            )
        # Pick random legal moves.
        actions: list[int] = []
        ref_moves: list[tuple[int, int, int]] = []
        for k, i in enumerate(active):
            legal_mvs = ref.legal_moves(states[i])
            mv = rng.choice(legal_mvs)
            ref_moves.append(mv)
            promo_arg = 0 if mv[2] in (0, 5) else mv[2]
            actions.append(encode_move(mv[0], mv[1], promo_arg, states[i].side_to_move))
        new_vs = vec.apply_action(vs, torch.tensor(actions, dtype=torch.long))
        new_vec_states = vec.to_states(new_vs)
        for k, i in enumerate(active):
            states[i] = ref.apply_move(states[i], ref_moves[k])
            assert states[i].to_fen() == new_vec_states[k].to_fen()
