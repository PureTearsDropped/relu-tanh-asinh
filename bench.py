"""Benchmark for fg-net, with the baselines that make the numbers mean something.

All models are stacked across seeds in a leading dimension, so `--seeds 16`
costs little more than `--seeds 1`. Gradient clipping is per-seed, so the
models stay independent.

    python bench.py --depth 32 --seeds 8 --model swiglu,gelu,fg
    python bench.py --diagnose          # linear / one-nonlinearity / ablation

0BSD.
"""
import argparse, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIN, DOUT = 32, 8


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


pos = lambda r: F.softplus(r) + 1e-4
inv = lambda v: math.log(math.expm1(max(v - 1e-4, 1e-6)))
f_ab = lambda x, a, b: torch.where(x >= 0, a * x, b * torch.tanh(a * x / b))
g_AB = lambda y, A, B: torch.where(y >= 0, y / A, (B / A) * torch.asinh(y / B))


class Teacher(nn.Module):
    """Multiplicative interaction: needs depth, has no ceiling, 1-R^2 = 0.998."""
    def __init__(self, width=128):
        super().__init__()
        self.A = nn.Parameter(torch.randn(width, DIN) * DIN ** -0.5)
        self.B = nn.Parameter(torch.randn(width, DIN) * DIN ** -0.5)
        self.o = nn.Linear(width, DOUT)

    def forward(self, x):
        return self.o(torch.tanh(x @ self.A.t()) * torch.tanh(x @ self.B.t()))


def _gens(S, seed0):
    return [torch.Generator().manual_seed(1000 + seed0 + i) for i in range(S)]


def _u(shape, fan, g):
    return (torch.rand(shape, generator=g) * 2 - 1) * (fan ** -0.5)


class _Stack(nn.Module):
    """Shared input/output projections for the stacked models."""
    def __init__(self, S, W, gen):
        super().__init__()
        self.Wi = nn.Parameter(torch.stack([_u((W, DIN), DIN, g) for g in gen]))
        self.bi = nn.Parameter(torch.stack([_u((W,), DIN, g) for g in gen]))
        self.Wo = nn.Parameter(torch.stack([_u((DOUT, W), W, g) for g in gen]))
        self.bo = nn.Parameter(torch.stack([_u((DOUT,), W, g) for g in gen]))

    def head(self, x):
        return torch.einsum('bi,sni->sbn', x, self.Wi) + self.bi[:, None, :]

    def tail(self, h):
        return torch.einsum('sbn,son->sbo', h, self.Wo) + self.bo[:, None, :]


class MLP(_Stack):
    def __init__(self, S, W=64, depth=4, seed0=0, act="gelu", scaled=True):
        gen = _gens(S, seed0); super().__init__(S, W, gen)
        self.d, self.act = depth, act
        sc = W ** -0.5 * (depth ** -0.5 if scaled else 1.0)
        self.Wm = nn.Parameter(torch.stack(
            [torch.randn(depth, W, W, generator=g) * sc for g in gen]))
        self.bm = nn.Parameter(torch.zeros(S, depth, W))

    def forward(self, x):
        h = self.head(x)
        act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[self.act]
        for L in range(self.d):
            h = h + torch.bmm(act(h), self.Wm[:, L].transpose(1, 2)) + self.bm[:, L][:, None, :]
        return self.tail(h)


class SwiGLU(_Stack):
    def __init__(self, S, W=64, depth=4, seed0=0, m=None, scaled=True):
        gen = _gens(S, seed0); super().__init__(S, W, gen)
        self.d = depth; M = m or max(1, W // 3)     # m = W//3 matches parameters
        sd = M ** -0.5 * (depth ** -0.5 if scaled else 1.0)
        st = lambda sh, s: nn.Parameter(torch.stack(
            [torch.randn(sh, generator=g) * s for g in gen]))
        self.Wg = st((depth, M, W), W ** -0.5)
        self.Wu = st((depth, M, W), W ** -0.5)
        self.Wd = st((depth, W, M), sd)
        self.bm = nn.Parameter(torch.zeros(S, depth, W))

    def forward(self, x):
        h = self.head(x)
        for L in range(self.d):
            gt = torch.bmm(h, self.Wg[:, L].transpose(1, 2))
            up = torch.bmm(h, self.Wu[:, L].transpose(1, 2))
            h = h + torch.bmm(F.silu(gt) * up, self.Wd[:, L].transpose(1, 2)) \
                  + self.bm[:, L][:, None, :]
        return self.tail(h)


class FG(_Stack):
    def __init__(self, S, W=64, depth=4, seed0=0, b0=0.1, scaled=True, ablate=False):
        gen = _gens(S, seed0); super().__init__(S, W, gen)
        self.d, self.ablate = depth, ablate
        ia, ib = inv(1.0), inv(b0)
        mk = lambda v: nn.Parameter(torch.full((S, depth, W), v))
        self.a, self.b, self.A, self.B = mk(ia), mk(ib), mk(ia), mk(ib)
        self.a0 = nn.Parameter(torch.full((S, W), ia))
        self.b0 = nn.Parameter(torch.full((S, W), ib))
        self.mu = nn.Parameter(torch.stack(
            [torch.randn(depth, W, W, generator=g) * 0.5 for g in gen]))
        self.dg = nn.Parameter(torch.full(
            (S, depth, W), -0.5 * math.log(depth) if scaled else 0.0))

    def forward(self, x):
        h = f_ab(self.head(x), pos(self.a0)[:, None, :], pos(self.b0)[:, None, :])
        for L in range(self.d):
            if self.ablate:                       # C = 0: the residual alone
                continue
            A, B = pos(self.A[:, L])[:, None, :], pos(self.B[:, L])[:, None, :]
            a, b = pos(self.a[:, L])[:, None, :], pos(self.b[:, L])[:, None, :]
            mu = self.mu[:, L]
            c = mu / (mu.abs().sum(-1, keepdim=True) + 1e-8)
            u = torch.bmm(g_AB(h, A, B), c.transpose(1, 2))
            h = h + f_ab(u, a, b) * torch.exp(self.dg[:, L].clamp(-6, 4))[:, None, :]
        return self.tail(h)


class Linear(_Stack):
    """Baseline 1: no nonlinearity at all."""
    def __init__(self, S, W=64, depth=1, seed0=0, **kw):
        super().__init__(S, W, _gens(S, seed0))
    def forward(self, x): return self.tail(self.head(x))


class OneNonlin(_Stack):
    """Baseline 2: one f, no depth."""
    def __init__(self, S, W=64, depth=1, seed0=0, b0=0.1, **kw):
        super().__init__(S, W, _gens(S, seed0))
        self.a = nn.Parameter(torch.full((S, W), inv(1.0)))
        self.b = nn.Parameter(torch.full((S, W), inv(b0)))
    def forward(self, x):
        return self.tail(f_ab(self.head(x), pos(self.a)[:, None, :], pos(self.b)[:, None, :]))


MODELS = {
    "linear":   lambda S, **k: Linear(S, **k),
    "onenl":    lambda S, **k: OneNonlin(S, **k),
    "relu":     lambda S, **k: MLP(S, act="relu", **k),
    "gelu":     lambda S, **k: MLP(S, act="gelu", **k),
    "silu":     lambda S, **k: MLP(S, act="silu", **k),
    "swiglu":   lambda S, **k: SwiGLU(S, **k),
    "swiglu_w": lambda S, **k: SwiGLU(S, m=k.get("W", 64), **k),
    "fg":       lambda S, **k: FG(S, **k),
    "fg_ablate": lambda S, **k: FG(S, ablate=True, **k),
}


def clip_per_seed(model, S, maxnorm=5.0):
    """Clip each stacked model separately; a global norm would couple them."""
    for p in model.parameters():
        if p.grad is None: continue
        g = p.grad.reshape(S, -1)
        g.mul_((maxnorm / (g.norm(dim=1, keepdim=True) + 1e-6)).clamp(max=1.0))


def run(name, S, depth, steps, width, scaled, dev):
    set_seed(1000); T = Teacher().to(dev).eval()
    for p in T.parameters(): p.requires_grad_(False)
    set_seed(0)
    m = MODELS[name](S, W=width, depth=depth, scaled=scaled).to(dev)
    npar = sum(p.numel() for p in m.parameters()) // S
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x = torch.randn(512, DIN, device=dev)
        with torch.no_grad(): y = T(x)
        opt.zero_grad(set_to_none=True)
        ((m(x) - y[None]) ** 2).mean(dim=(1, 2)).sum().backward()
        clip_per_seed(m, S); opt.step(); sch.step()
    with torch.no_grad():
        x = torch.randn(20000, DIN, device=dev); y = T(x)
        loss = ((m(x) - y[None]) ** 2).mean(dim=(1, 2)) / y.var()
    return [float(v) for v in loss], npar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swiglu,gelu,fg")
    ap.add_argument("--depth", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--no-scale", action="store_true",
                    help="omit the 1/sqrt(L) residual-branch init (see RESULTS 3)")
    ap.add_argument("--diagnose", action="store_true",
                    help="run the baselines that decide whether a task is usable")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    names = ["linear", "onenl", "gelu", "fg_ablate"] if a.diagnose \
        else a.model.split(",")
    print(f"depth {a.depth}  width {a.width}  {a.steps} steps  {a.seeds} seeds  "
          f"residual init {'1/sqrt(L)' if not a.no_scale else 'plain'}  [{dev}]")
    print(f"{'model':>12} {'loss':>10} {'std':>9} {'params':>10}")
    print("-" * 46)
    res = {}
    for n in names:
        d = 1 if n in ("linear", "onenl") else a.depth
        L, npar = run(n, a.seeds, d, a.steps, a.width, not a.no_scale, dev)
        res[n] = np.mean(L)
        print(f"{n:>12} {np.mean(L):10.5f} {np.std(L, ddof=1):9.5f} {npar:10,}")
    if a.diagnose:
        v = res["onenl"] / res["gelu"]
        print(f"\ndepth value = one-nonlinearity / deep = {v:.2f}"
              f"   {'OK' if v >= 1.5 else 'TOO LOW - this task does not need depth'}")
        print(f"ablation (C=0) {res['fg_ablate']:.5f} must be clearly worse than any real model")


if __name__ == "__main__":
    main()
