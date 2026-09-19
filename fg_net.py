"""fg-net: a conjugate-pair activation/synapse layer.

    h <- h + d * f_{a,b}( C @ g_{A,B}(h) )

where f saturates on the negative side and g is *not* its exact inverse.
The mismatch is what produces the nonlinearity; see SPEC.md section 2.

0BSD.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["f_ab", "g_AB", "FGLayer", "FGNet", "ternary", "binary"]

_pos = lambda r: F.softplus(r) + 1e-4
_inv = lambda v: math.log(math.expm1(max(v - 1e-4, 1e-6)))


def f_ab(x, a, b):
    """Neuron. Linear for x>=0, saturates to -b for x<0. C^2, not C^3.

    Use torch.where, not a ReLU rewrite: relu'(0)=0 in PyTorch makes the
    gradient wrong at exactly x=0 (the true value is f'(0)=a).
    """
    return torch.where(x >= 0, a * x, b * torch.tanh(a * x / b))


def g_AB(y, A, B):
    """Decoder. Defined on all of R; grows like log on the negative side.

    Not the exact inverse of f_ab -- that is deliberate. With the exact
    inverse, consecutive layers telescope and all interior nonlinearity
    cancels.
    """
    return torch.where(y >= 0, y / A, (B / A) * torch.asinh(y / B))


def binary(mu, scale):
    """BinaryConnect: hard sign forward, smooth surrogate backward."""
    hard = torch.sign(mu)
    hard = torch.where(hard == 0, torch.ones_like(hard), hard)
    soft = torch.tanh(3.0 * mu)
    return scale * (hard.detach() + soft - soft.detach())


def ternary(mu, scale, keep):
    """Ternary {-s, 0, +s}. `keep` is a 0/1 mask (see FGLayer.coupling)."""
    hard = torch.sign(mu) * keep
    soft = torch.tanh(3.0 * mu)
    return scale * (hard.detach() + soft - soft.detach())


class FGLayer(nn.Module):
    """One residual f-g block.

    quant: None (real, L1-normalised) | "ternary" | "binary"
    frac:  fraction of couplings kept per row when quant="ternary"
    depth: total network depth; the residual branch is initialised at
           1/sqrt(depth), which matters a great deal (RESULTS.md 3).
    """

    def __init__(self, n, b0=0.1, quant=None, frac=0.5, depth=1):
        super().__init__()
        ia, ib = _inv(1.0), _inv(b0)
        self.a = nn.Parameter(torch.full((n,), ia))
        self.b = nn.Parameter(torch.full((n,), ib))
        self.A = nn.Parameter(torch.full((n,), ia))
        self.B = nn.Parameter(torch.full((n,), ib))
        self.mu = nn.Parameter(torch.randn(n, n) * 0.5)
        self.dg = nn.Parameter(torch.full((n,), -0.5 * math.log(depth)))
        self.n, self.quant, self.frac = n, quant, frac
        self.scale = n ** -0.5

    def coupling(self, v=None):
        """v: diagonal curvature estimate, e.g. Adam's exp_avg_sq.

        With v, weights are ranked by sqrt(v)*|mu| -- the second-order
        estimate of how much zeroing a weight costs. Untested; see SPEC 4.4.
        """
        if self.quant is None:
            return self.mu / (self.mu.abs().sum(-1, keepdim=True) + 1e-8)
        if self.quant == "binary":
            return binary(self.mu, self.scale)
        imp = (v.sqrt() * self.mu.abs()) if v is not None else self.mu.abs()
        k = max(1, int(self.n * self.frac))
        kth = imp.topk(k, dim=-1).values[..., -1:]
        return ternary(self.mu, self.scale, (imp >= kth).float())

    def forward(self, h, v=None, sigma=0.0):
        A, B = _pos(self.A), _pos(self.B)
        a, b = _pos(self.a), _pos(self.b)
        c = self.coupling(v)
        if sigma > 0 and self.training:
            c = c * (1 + sigma * torch.randn_like(c))
        z = f_ab(g_AB(h, A, B) @ c.t(), a, b)
        d = (1.0 / (self.scale * math.sqrt(self.n))) if self.quant \
            else torch.exp(self.dg.clamp(-6, 4))
        return h + z * d


class FGNet(nn.Module):
    def __init__(self, d_in, d_out, width=64, depth=4, b0=0.1, **kw):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.a0 = nn.Parameter(torch.full((width,), _inv(1.0)))
        self.b0 = nn.Parameter(torch.full((width,), _inv(b0)))
        self.layers = nn.ModuleList(
            [FGLayer(width, b0=b0, depth=depth, **kw) for _ in range(depth)])
        self.out = nn.Linear(width, d_out)

    def forward(self, x, sigma=0.0):
        h = f_ab(self.inp(x), _pos(self.a0), _pos(self.b0))
        for layer in self.layers:
            h = layer(h, sigma=sigma)
        return self.out(h)
