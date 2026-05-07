"""Statistical parity tests for triton_rollout vs vec MCTS."""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("triton_rollout requires CUDA", allow_module_level=True)

from chessvec.action_encoding import ACTION_SIZE  # noqa: E402
from chessvec.reference import State  # noqa: E402
from chessvec.vectorized import from_states, legal_action_mask  # noqa: E402
from chessvec.triton_step import triton_rollout  # noqa: E402


_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
]


def test_root_actions_only_legal() -> None:
    """Every sampled root action must be a legal action in the root state."""
    for fen in _FENS:
        state = State.from_fen(fen)
        vs = from_states([state] * 1024, device="cuda")
        legal = legal_action_mask(vs)  # [B, ACTION_SIZE]
        ra, _ = triton_rollout(vs, depth=8, seed=0)
        # Every root_action must satisfy legal[b, ra[b]] == True.
        ok = legal.gather(1, ra.view(-1, 1)).squeeze(1)
        assert ok.all(), f"illegal root action in {fen}: {(~ok).sum().item()} of {ok.numel()}"


def test_root_action_distribution_matches_uniform() -> None:
    """At depth=1 the root action distribution should be (approximately)
    uniform over the legal moves at the root."""
    state = State.from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    B = 8192
    vs = from_states([state] * B, device="cuda")
    ra, _ = triton_rollout(vs, depth=1, seed=0)

    # Histogram over actions; only the legal ones at root should have hits.
    hits = torch.zeros(ACTION_SIZE, dtype=torch.long, device="cuda")
    hits.scatter_add_(0, ra, torch.ones_like(ra))
    legal0 = legal_action_mask(vs[:1] if hasattr(vs, "__getitem__") else vs)
    # Use the (identical) legal mask of any row.
    legal_row = legal_action_mask(vs)[0]
    n_legal = int(legal_row.sum().item())
    expected = B / n_legal

    # All hits should be on legal actions; no hits on illegal.
    assert (hits[~legal_row] == 0).all()
    legal_hits = hits[legal_row]
    # Chi-square-ish sanity: each legal action seen, max/min within ~3 sigma.
    sigma = expected ** 0.5
    assert legal_hits.min().item() > expected - 5 * sigma, (
        f"min hit {legal_hits.min().item()} below expected {expected:.1f} - 5sigma"
    )
    assert legal_hits.max().item() < expected + 5 * sigma, (
        f"max hit {legal_hits.max().item()} above expected {expected:.1f} + 5sigma"
    )


def test_terminal_at_root_returns_signed_leaf() -> None:
    """A position that is checkmate at the root should yield leaf_value = -1
    (mover loses; root_player == mover -> -1 from root POV)."""
    # Fool's mate: 1.f3 e5 2.g4 Qh4#. Black has just checkmated white; white
    # to move with no legal moves and king in check.
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    state = State.from_fen(fen)
    vs = from_states([state] * 8, device="cuda")
    legal = legal_action_mask(vs)
    assert not legal.any(dim=1).any(), "test FEN should be checkmate (no legal moves)"

    ra, lv = triton_rollout(vs, depth=4, seed=0)
    # Mover (white) loses; root_player is white -> leaf_value = -1.
    assert (lv == -1.0).all(), lv.tolist()
