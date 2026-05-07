"""Microbenchmark just for the Triton legal_action_mask kernel.

Run as: uv run python benchmarks/bench_legal_mask.py
"""
from __future__ import annotations

import time
import torch

from chessvec.reference import State
from chessvec.vectorized import from_states, legal_action_mask
from chessvec.triton_step import triton_legal_action_mask

POSITIONS = {
    "startpos":  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwipete":  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "midgame":   "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
    "endgame_r": "8/8/8/4k3/8/8/4K3/R7 w - - 0 1",
    "ep_pos":    "rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 2",
}

def time_kernel(vs, n=1500):
    torch.cuda.synchronize()
    # warmup
    for _ in range(80):
        triton_legal_action_mask(vs)
    torch.cuda.synchronize()
    # take min of 3 short runs to control for noise
    best = 1e9
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            triton_legal_action_mask(vs)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / n)
    return best


def main():
    B = 4096
    print(f"B={B}, time per call (us):")
    print(f"{'position':<12}  {'us':>10}  {'us/board':>10}")
    for name, fen in POSITIONS.items():
        st = State.from_fen(fen)
        vs = from_states([st] * B, device="cuda")
        t = time_kernel(vs)
        print(f"{name:<12}  {t*1e6:>10.1f}  {t*1e6/B:>10.4f}")


if __name__ == "__main__":
    main()
