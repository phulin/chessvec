"""Per-stage timing of vec[cuda] / triton[cuda] step loop, with a dummy
~5M-parameter CNN forward pass added (representing the AlphaZero-style
value/policy net evaluation that dominates real MCTS training).

Stages: NN forward, mask, reset, sample, apply.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from chessvec.reference import State
from chessvec.vectorized import (
    apply_action,
    from_states,
    legal_action_mask,
)
from chessvec.triton_step import triton_step

POSITIONS = {
    "kiwipete": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
}


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class DummyAZNet(nn.Module):
    """Roughly AlphaZero-shaped CNN: 13-plane input → C-channel residual
    tower → policy + value heads. Sized to ~5M params with C=128, N=17.
    """
    def __init__(self, in_planes: int = 13, c: int = 128, n_blocks: int = 17,
                 action_size: int = 4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(n_blocks)])
        self.pol_conv = nn.Conv2d(c, 73, 1, bias=False)   # 73 move planes / sq
        self.pol_bn = nn.BatchNorm2d(73)
        self.val_conv = nn.Conv2d(c, 1, 1, bias=False)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(64, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(self.stem(x))
        p = self.pol_bn(self.pol_conv(h)).reshape(x.size(0), -1)
        v = torch.relu(self.val_bn(self.val_conv(h))).reshape(x.size(0), -1)
        v = torch.tanh(self.val_fc2(torch.relu(self.val_fc1(v))))
        return p, v


def state_to_planes(vs) -> torch.Tensor:
    """[B, 64] int8 pieces → [B, 13, 8, 8] float planes (12 piece codes + side)."""
    pieces = vs.pieces  # [B, 64] int8 in 0..12
    B = pieces.size(0)
    planes = torch.zeros(B, 13, 64, dtype=torch.float16, device=pieces.device)
    for code in range(1, 13):
        planes[:, code - 1] = (pieces == code).to(torch.float16)
    planes[:, 12] = vs.side_to_move.view(B, 1).to(torch.float16)
    return planes.view(B, 13, 8, 8)


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


def time_loop(state, B, n_steps, use_triton, net):
    vs = from_states([state] * B, device="cuda")
    initial = vs.clone()
    initial_mask = legal_action_mask(initial)

    for _ in range(8):
        with torch.no_grad():
            net(state_to_planes(vs))
        mask = legal_action_mask(vs)
        need_reset = ~mask.any(dim=1)
        vs = _branchless_reset(vs, initial, need_reset)
        mask = torch.where(need_reset.view(B, 1), initial_mask, mask)
        action = torch.multinomial(mask.float(), num_samples=1).squeeze(1)
        vs = (triton_step if use_triton else apply_action)(vs, action)
    torch.cuda.synchronize()

    stages = ["nn", "mask", "reset", "sample", "apply"]
    starts = {s: [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)] for s in stages}
    ends = {s: [torch.cuda.Event(enable_timing=True) for _ in range(n_steps)] for s in stages}
    totals = {s: 0.0 for s in stages}

    for i in range(n_steps):
        starts["nn"][i].record()
        with torch.no_grad():
            _, _ = net(state_to_planes(vs))
        ends["nn"][i].record()

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
    return {s: totals[s] / n_steps for s in stages}


def main():
    import sys
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    net = DummyAZNet().to("cuda").to(torch.float16).eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Net params: {n_params/1e6:.2f}M (dtype=fp16)  torch.compile=on")
    mode = __import__("os").environ.get("COMPILE_MODE", "reduce-overhead")
    print(f"compile mode={mode}")
    net = torch.compile(net, mode=mode, fullgraph=True)
    print(f"B={B}, n_steps={n_steps}, time per step (us)")
    print(f"{'pos':<10} {'eng':<7} {'nn':>7} {'mask':>7} {'reset':>7} {'sample':>7} {'apply':>7} {'total':>8}")
    print("-" * 60)
    for name, fen in POSITIONS.items():
        st = State.from_fen(fen)
        for use_triton, tag in [(False, "vec"), (True, "triton")]:
            t = time_loop(st, B, n_steps, use_triton, net)
            total = sum(t.values())
            print(
                f"{name:<10} {tag:<7} {t['nn']*1000:>7.1f} {t['mask']*1000:>7.1f} "
                f"{t['reset']*1000:>7.1f} {t['sample']*1000:>7.1f} {t['apply']*1000:>7.1f} "
                f"{total*1000:>8.1f}"
            )


if __name__ == "__main__":
    main()
