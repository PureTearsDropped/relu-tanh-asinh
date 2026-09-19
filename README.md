# fg-net

A residual layer built from a **conjugate pair** — a saturating neuron `f` and a
decoder `g` that is deliberately *not* its inverse:

$$h \leftarrow h + d\,\cdot\,f_{a,b}\big(C\,g_{A,B}(h)\big)$$

$$f_{a,b}(x)=\begin{cases}ax & x\ge0\\ b\tanh(ax/b) & x<0\end{cases}
\qquad
g_{A,B}(y)=\begin{cases}y/A & y\ge0\\ (B/A)\,\mathrm{asinh}(y/B) & y<0\end{cases}$$

Both are $C^2$ but not $C^3$. The nonlinearity of the block is the **mismatch**
between $g$ and $f^{-1}$, and to leading order it is a cubic:

$$g_{a,b}\big(f_{a,b}(x)\big)-x=-\frac{a^2}{2b^2}x^3+O(x^5)$$

If $g$ *were* the exact inverse, consecutive layers would telescope and every
interior nonlinearity would cancel. The `asinh` decoder makes that impossible:
the third derivatives differ by $-3a^3/b^2$ for every $a,b$.

## What this repository is

An **honest record**, not a claim that the construction wins. Measured against
plain MLPs and SwiGLU on a task built so that depth genuinely matters:

| depth 32, 8 seeds, matched parameters | normalised MSE |
|---|---|
|SwiGLU|**0.07095 ± 0.00086**|
|MLP gelu|0.11926 ± 0.00128|
|**this construction**|0.12216 ± 0.00066|

**SwiGLU wins.** Two earlier conclusions in favour of this construction were
withdrawn after checking them: one because the benchmark could be solved without
any depth at all, one because a residual-branch initialisation had been applied
to the baseline's competitor but not to the baseline. Both are documented in
[RESULTS.md](RESULTS.md).

What did survive is more useful than the construction itself:

- **initialise the residual branch at $1/\sqrt{L}$.** This was worth 30-41% to
  plain MLPs at depth 64, and SwiGLU reaches NaN by depth 16 without it
- **the kink is what matters, not the curve.** Replacing `tanh` with `hardtanh`
  at inference is 8.6x faster and costs nothing measurable; removing the
  nonlinearity costs 0.23 accuracy
- **classification accuracy near its ceiling hides quantisation cost.** The same
  substitutions were free on MNIST and cost +27-45% on regression
- **a benchmark checklist** that would have caught both retractions
  ([SPEC.md](SPEC.md) section 8)

## Files

| | |
|---|---|
|[`fg_net.py`](fg_net.py)|reference implementation (~120 lines, PyTorch)|
|[`SPEC.md`](SPEC.md)|construction: definitions, derivatives, normalisation, quantisation, initialisation, diagnostics|
|[`RESULTS.md`](RESULTS.md)|measurements, including the two retracted conclusions and why|
|[`bench.py`](bench.py)|the benchmark, with all baselines|

## Quick start

```python
from fg_net import FGNet
model = FGNet(d_in=32, d_out=8, width=64, depth=32)          # real weights
model = FGNet(32, 8, width=64, depth=32, quant="ternary")    # ternary coupling
```

```bash
python bench.py --depth 32 --seeds 8 --model swiglu,gelu,fg
```

## Before you trust any number from this

Run the diagnostics in SPEC.md section 8. The short version: measure a **linear
baseline**, a **one-nonlinearity baseline**, and an **ablation with the interior
zeroed**, on every new task, before comparing anything. A benchmark that any of
those three can solve is not measuring what you think it is.

## Licence

0BSD.
