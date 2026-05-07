"""Per-stage timing of the vec[cuda] / triton[cuda] step loop.

Mirrors bench_step.bench_vectorized + bench_triton, but uses CUDA events to
measure each phase (mask compute, reset, multinomial, apply) per iteration.
"""
from __future__ import annotations

import time
import torch

from chessvec.reference import State
from chessvec.vectorized import (
    apply_action,
    from_states,
    legal_action_mask,
)
from chessvec.triton_step import triton_step

POSITIONS = {
    "startpos":  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "kiwipete":  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "midgame":   "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 8",
}


def _branchless_reset(vs, initial, need_reset):
    from chessvec.vectorized import VState
    m = need_reset
    return VState(
        pieces=torch.where(m.unsqueeze(1), initial.pieces, vs.pieces),
        side_to_move=torch.where(m, initial.side_to_move, vs.side_to_move),
        castling=torch.where(m, initial.castling, vs.castling),
        en_passant=torch.where(m, initial.en_passant, vs.en_passant),
        halfmove_clock=torch.where(m, initial.halfmove_clock, vs.halfmove_clock),
        fullmove_number=torch.where(m, initial.fullmove_number, vs.fullmove_number),
    )


def time_loop(state, B, n_steps, use_triton):
    vs = from_states([state] * B, device="cuda")
    initial = vs.clone()
    initial_mask = legal_action_mask(initial)

    # warmup
    for _ in range(20):
        mask = legal_action_mask(vs)
        need_reset = ~mask.any(dim=1)
        vs = _branchless_reset(vs, initial, need_reset)
        mask = torch.where(need_reset.view(B, 1), initial_mask, mask)
        action = torch.multinomial(mask.float(), num_samples=1).squeeze(1)
        vs = (triton_step if use_triton else apply_action)(vs, action)
    torch.cuda.synchronize()

    # Use CUDA events for fine-grained timing.
    stages = ["mask", "reset", "sample", "apply"]
    totals = {s: 0.0 for s in stages}
    n_events = n_steps + 1
    starts = {s: [torch.cuda.Event(enable_timing=True) for _ in range(n_events)] for s in stages}
    ends = {s: [torch.cuda.Event(enable_timing=True) for _ in range(n_events)] for s in stages}

    for i in range(n_steps):
        starts["mask"][i].record()
        mask = legal_action_mask(vs)
        ends["mask"][i].record()

        starts["reset"][i].record()
        need_reset = ~mask.any(dim=1)
        vs = _branchless_reset(vs, initial, need_reset)
        mask2 = torch.where(need_reset.view(B, 1), initial_mask, mask)
        ends["reset"][i].record()

        starts["sample"][i].record()
        action = torch.multinomial(mask2.float(), num_samples=1).squeeze(1)
        ends["sample"][i].record()

        starts["apply"][i].record()
        vs = (triton_step if use_triton else apply_action)(vs, action)
        ends["apply"][i].record()

    torch.cuda.synchronize()
    for s in stages:
        for i in range(n_steps):
            totals[s] += starts[s][i].elapsed_time(ends[s][i])  # ms
    return {s: totals[s] / n_steps for s in stages}  # ms per step


def main():
    B = 4096
    n_steps = 200
    print(f"B={B}, n_steps={n_steps}, time per step (us)")
    print(f"{'pos':<12} {'engine':<8} {'mask':>8} {'reset':>8} {'sample':>8} {'apply':>8} {'total':>8}")
    print("-" * 58)
    for name, fen in POSITIONS.items():
        st = State.from_fen(fen)
        for use_triton, tag in [(False, "vec"), (True, "triton")]:
            t = time_loop(st, B, n_steps, use_triton)
            total = sum(t.values())
            print(
                f"{name:<12} {tag:<8} {t['mask']*1000:>8.1f} {t['reset']*1000:>8.1f} "
                f"{t['sample']*1000:>8.1f} {t['apply']*1000:>8.1f} {total*1000:>8.1f}"
            )


if __name__ == "__main__":
    main()
