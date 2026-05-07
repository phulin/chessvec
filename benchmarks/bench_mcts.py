"""MCTS-rollout throughput benchmark.

Implements flat-Monte-Carlo / root-parallel MCTS: from a given root state,
each "simulation" is

    1. pick a uniformly-random legal root action (the action under evaluation),
    2. random-rollout for up to `depth` plies,
    3. score the leaf via :func:`game_result` from the root player's POV
       (+1 win / -1 loss / 0 draw / 0 if non-terminal).

Per-action value estimates are aggregated and the argmax is the recommended
move. We don't care about the chosen move here — it's just to make the
backprop work realistic.

Engines compared per position:
- reference (Python loop, B simulations done sequentially)
- vec[cpu]    -- VState batched on CPU
- vec[cuda]   -- VState batched on CUDA
- triton[cuda] -- vec[cuda] but with the Triton single-kernel apply_action

Run with:
    uv run python benchmarks/bench_mcts.py
    uv run python benchmarks/bench_mcts.py --batches 64,256,1024 --depth 40
"""

from __future__ import annotations

import argparse
import time

import torch

from chessvec.action_encoding import ACTION_SIZE
from chessvec.reference import State, apply_move, legal_moves
from chessvec.vectorized import (
    DRAW,
    WHITE,
    WHITE_WIN,
    VState,
    apply_action,
    from_states,
    game_result,
    legal_action_mask,
)

try:
    from chessvec.triton_step import _HAS_TRITON, triton_rollout, triton_step
except ImportError:  # pragma: no cover
    _HAS_TRITON = False
    triton_step = None  # type: ignore[assignment]
    triton_rollout = None  # type: ignore[assignment]


POSITIONS: dict[str, str] = {
    "startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "midgame": "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
    "endgame_rook": "8/8/8/4k3/8/8/4K3/R7 w - - 0 1",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _default_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _branchless_reset(vs: VState, initial: VState, need_reset: torch.Tensor) -> VState:
    m = need_reset
    return VState(
        pieces=torch.where(m.unsqueeze(1), initial.pieces, vs.pieces),
        side_to_move=torch.where(m, initial.side_to_move, vs.side_to_move),
        castling=torch.where(m, initial.castling, vs.castling),
        en_passant=torch.where(m, initial.en_passant, vs.en_passant),
        halfmove_clock=torch.where(m, initial.halfmove_clock, vs.halfmove_clock),
        fullmove_number=torch.where(m, initial.fullmove_number, vs.fullmove_number),
    )


# ---------------------------------------------------------------------------
# Reference (single-state Python) MCTS.
# ---------------------------------------------------------------------------


def bench_reference(state: State, n_sims: int, depth: int) -> float:
    rng = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()
    for _ in range(n_sims):
        cur = state
        for _ in range(depth):
            moves = legal_moves(cur)
            if not moves:
                break
            idx = int(torch.randint(0, len(moves), (1,), generator=rng).item())
            cur = apply_move(cur, moves[idx])
            if cur.halfmove_clock >= 100:
                break
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Vectorized MCTS (one simulation per batch slot).
# ---------------------------------------------------------------------------


def _run_vec_mcts(
    state: State,
    batch_size: int,
    depth: int,
    device: torch.device,
    use_triton: bool,
) -> tuple[float, torch.Tensor]:
    """Returns (elapsed_seconds, per-action visit counts [ACTION_SIZE]).

    Each batch slot is one simulation: at step 0 pick a uniformly-random legal
    root action and remember it. At each subsequent step pick a uniformly-
    random legal action. After `depth` plies (or terminal earlier) score the
    leaf via game_result and accumulate (visits, value-sum) per root action.
    """
    apply_fn = triton_step if use_triton else apply_action
    assert apply_fn is not None

    initial = from_states([state] * batch_size, device=device)
    initial_mask = legal_action_mask(initial)
    root_player = int(state.side_to_move)

    rng = torch.Generator(device="cpu").manual_seed(0) if device.type == "cpu" else None

    visits = torch.zeros(ACTION_SIZE, dtype=torch.float32, device=device)
    value_sum = torch.zeros(ACTION_SIZE, dtype=torch.float32, device=device)

    _sync(device)
    t0 = time.perf_counter()

    vs = initial.clone()
    # done[b] = True once env b reached terminal during this rollout.
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    # leaf_value[b] = +1/-1/0 once terminal reached (from root_player POV).
    leaf_value = torch.zeros(batch_size, dtype=torch.float32, device=device)
    # root_action[b] = the action taken at depth 0 in env b.
    root_action = torch.zeros(batch_size, dtype=torch.long, device=device)

    for d in range(depth):
        mask = legal_action_mask(vs)
        # Envs hitting terminal this step get scored & frozen.
        no_move = ~mask.any(dim=1)
        new_done = no_move & ~done
        if new_done.any():
            res = game_result(vs)  # [B] long with WHITE_WIN/BLACK_WIN/DRAW/ONGOING
            # game_result returns ONGOING when has_move; here we only care about
            # rows with no_move, which will be terminal.
            white_pov = torch.where(
                res == WHITE_WIN,
                torch.tensor(1.0, device=device),
                torch.where(
                    res == DRAW,
                    torch.tensor(0.0, device=device),
                    torch.tensor(-1.0, device=device),
                ),
            )
            sign = 1.0 if root_player == WHITE else -1.0
            v = white_pov * sign
            leaf_value = torch.where(new_done, v, leaf_value)
            done = done | new_done
            # keep batch valid by resetting to initial; subsequent steps for
            # these rows are wasted work but no host sync needed.
            vs = _branchless_reset(vs, initial, new_done)
            mask = torch.where(new_done.view(batch_size, 1), initial_mask, mask)

        # Sample a uniformly-random legal action per env.
        action = torch.multinomial(mask.float(), num_samples=1, generator=rng).squeeze(1)
        if d == 0:
            root_action = action.clone()

        vs = apply_fn(vs, action)

    _sync(device)
    elapsed = time.perf_counter() - t0

    # Backprop into per-root-action stats. Non-terminal leaves contribute 0.
    visits.scatter_add_(0, root_action, torch.ones_like(leaf_value))
    value_sum.scatter_add_(0, root_action, leaf_value)
    # Touch the result so the work isn't dead-code-eliminated by anyone.
    _ = (visits, value_sum)
    return elapsed, visits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", default=None)
    p.add_argument("--batches", default="16,64,256,1024,4096")
    p.add_argument("--depth", type=int, default=30, help="rollout depth in plies")
    p.add_argument(
        "--ref-sims",
        type=int,
        default=200,
        help="reference simulations per position (small; reference is slow)",
    )
    p.add_argument("--positions", default=",".join(POSITIONS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device_names = (
        [d for d in args.devices.split(",") if d] if args.devices else _default_devices()
    )
    devices = [torch.device(d) for d in device_names]
    batches = [int(x) for x in args.batches.split(",") if x]
    names = [n for n in args.positions.split(",") if n]

    print(
        f"devices={[str(d) for d in devices]}  depth={args.depth}  "
        f"ref_sims={args.ref_sims}  batches={batches}"
    )
    print()

    header = f"{'position':<14} {'engine':<22} {'sims/s':>12} {'plies/s':>14} {'wall (s)':>10}"
    print(header)
    print("-" * len(header))

    for name in names:
        fen = POSITIONS[name]
        state = State.from_fen(fen)

        # Reference: warmup + measure.
        bench_reference(state, n_sims=4, depth=args.depth)
        elapsed = bench_reference(state, n_sims=args.ref_sims, depth=args.depth)
        sps = args.ref_sims / elapsed
        plies_s = args.ref_sims * args.depth / elapsed
        print(f"{name:<14} {'reference':<22} {sps:>12,.1f} {plies_s:>14,.0f} {elapsed:>10.3f}")

        for device in devices:
            for B in batches:
                # warmup
                _run_vec_mcts(state, B, args.depth, device, use_triton=False)
                elapsed, _ = _run_vec_mcts(state, B, args.depth, device, use_triton=False)
                sps = B / elapsed
                plies_s = B * args.depth / elapsed
                tag = f"vec[{device.type}] B={B}"
                print(f"{name:<14} {tag:<22} {sps:>12,.1f} {plies_s:>14,.0f} {elapsed:>10.3f}")

                if _HAS_TRITON and device.type == "cuda":
                    _run_vec_mcts(state, B, args.depth, device, use_triton=True)
                    elapsed, _ = _run_vec_mcts(
                        state, B, args.depth, device, use_triton=True
                    )
                    sps = B / elapsed
                    plies_s = B * args.depth / elapsed
                    tag = f"triton[{device.type}] B={B}"
                    print(
                        f"{name:<14} {tag:<22} {sps:>12,.1f} {plies_s:>14,.0f} {elapsed:>10.3f}"
                    )

                    # Persistent fused-rollout kernel.
                    initial = from_states([state] * B, device=device)
                    triton_rollout(initial, depth=args.depth, seed=0)  # warmup
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    triton_rollout(initial, depth=args.depth, seed=1)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    sps = B / elapsed
                    plies_s = B * args.depth / elapsed
                    tag = f"fused[{device.type}] B={B}"
                    print(
                        f"{name:<14} {tag:<22} {sps:>12,.1f} {plies_s:>14,.0f} {elapsed:>10.3f}"
                    )
        print()


if __name__ == "__main__":
    main()
