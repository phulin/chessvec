"""Step-throughput benchmark: reference vs vectorized rules engines.

For each named position we measure how many env-steps per second each engine
sustains. A "step" is: compute the legal action mask, pick one legal action,
apply it. The vectorized engine is benchmarked at several batch sizes; the
reference engine is run B times in a Python loop for the same total work.

Run with:
    uv run python benchmarks/bench_step.py
    uv run python benchmarks/bench_step.py --device cuda --batches 1024,4096
"""

from __future__ import annotations

import argparse
import time

import torch

from chessvec.reference import State, apply_move, legal_moves
from chessvec.vectorized import (
    apply_action,
    from_states,
    legal_action_mask,
)

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _default_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


POSITIONS: dict[str, str] = {
    "startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "midgame": "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
    "endgame_kpk": "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "endgame_rook": "8/8/8/4k3/8/8/4K3/R7 w - - 0 1",
    "tactical": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
}


def bench_reference(state: State, n_steps: int) -> float:
    """Run n_steps random-legal steps starting from `state`. Returns elapsed s.

    Resets to `state` whenever a terminal is reached so we keep stepping.
    """
    rng = torch.Generator().manual_seed(0)
    cur = state
    t0 = time.perf_counter()
    for _ in range(n_steps):
        moves = legal_moves(cur)
        if not moves:
            cur = state
            continue
        idx = int(torch.randint(0, len(moves), (1,), generator=rng).item())
        cur = apply_move(cur, moves[idx])
        # Cheap terminal check: hit the 50-move clock or no-king edge cases by
        # just resetting periodically. For benchmarking we don't care about
        # game endings; we just want representative work.
        if cur.halfmove_clock >= 100:
            cur = state
    return time.perf_counter() - t0


def bench_vectorized(
    state: State, batch_size: int, n_steps: int, device: torch.device
) -> tuple[float, float]:
    """Run n_steps vectorized batched steps from `state` replicated B-wide.

    Returns (elapsed_seconds, sync_overhead_seconds).
    """
    vs = from_states([state] * batch_size, device=device)
    # multinomial accepts a CPU generator only when the input is CPU-resident;
    # on accelerators we fall back to the global RNG (good enough for timing).
    rng = torch.Generator(device="cpu").manual_seed(0) if device.type == "cpu" else None
    initial = vs.clone()

    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        mask = legal_action_mask(vs)
        # Reset terminal envs back to the initial state before sampling, so
        # every row of `mask` we sample from has at least one legal action.
        any_legal = mask.any(dim=1)
        if not bool(any_legal.all()):
            idx = (~any_legal).nonzero(as_tuple=True)[0]
            vs.pieces[idx] = initial.pieces[idx]
            vs.side_to_move[idx] = initial.side_to_move[idx]
            vs.castling[idx] = initial.castling[idx]
            vs.en_passant[idx] = initial.en_passant[idx]
            vs.halfmove_clock[idx] = initial.halfmove_clock[idx]
            vs.fullmove_number[idx] = initial.fullmove_number[idx]
            mask = legal_action_mask(vs)
        action = torch.multinomial(mask.float(), num_samples=1, generator=rng).squeeze(1)
        vs = apply_action(vs, action)
    sync_t0 = time.perf_counter()
    _sync(device)
    sync_overhead = time.perf_counter() - sync_t0 if device.type != "cpu" else 0.0
    return time.perf_counter() - t0, sync_overhead


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--devices",
        default=None,
        help="comma-separated devices (default: cpu plus any available accelerator)",
    )
    p.add_argument(
        "--batches",
        default="1,16,64,256,1024",
        help="comma-separated vectorized batch sizes to benchmark",
    )
    p.add_argument(
        "--ref-steps",
        type=int,
        default=2000,
        help="reference engine steps per position (slow; ~1k-10k is plenty)",
    )
    p.add_argument(
        "--vec-steps",
        type=int,
        default=200,
        help="vectorized engine outer iterations per position (per batch size)",
    )
    p.add_argument(
        "--positions",
        default=",".join(POSITIONS),
        help="comma-separated subset of positions to benchmark",
    )
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
        f"devices={[str(d) for d in devices]}  ref_steps={args.ref_steps}  "
        f"vec_steps={args.vec_steps}"
    )
    print()

    header = f"{'position':<14} {'engine':<22} {'env-steps/s':>14} {'wall (s)':>10}"
    print(header)
    print("-" * len(header))

    for name in names:
        fen = POSITIONS[name]
        state = State.from_fen(fen)

        # Reference: warmup + measure.
        bench_reference(state, n_steps=64)
        elapsed = bench_reference(state, n_steps=args.ref_steps)
        sps = args.ref_steps / elapsed
        print(f"{name:<14} {'reference':<22} {sps:>14,.0f} {elapsed:>10.3f}")

        for device in devices:
            for B in batches:
                bench_vectorized(state, batch_size=B, n_steps=4, device=device)  # warmup
                elapsed, sync = bench_vectorized(
                    state, batch_size=B, n_steps=args.vec_steps, device=device
                )
                env_steps = B * args.vec_steps
                sps = env_steps / elapsed
                tag = f"vec[{device.type}] B={B}"
                extra = f" (sync {sync * 1000:.1f}ms)" if sync else ""
                print(f"{name:<14} {tag:<22} {sps:>14,.0f} {elapsed:>10.3f}{extra}")
        print()


if __name__ == "__main__":
    main()
