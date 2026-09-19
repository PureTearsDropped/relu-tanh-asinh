# Specification

`[verified]` = checked numerically. `[untested]` = designed but not measured.

## 1. The two functions

### 1.1 Neuron `f`

$$f_{a,b}(x)=\begin{cases}a\,x & x\ge 0\\[4pt] b\,\tanh\!\big(\tfrac{a x}{b}\big) & x<0\end{cases}\qquad a,b>0$$

```python
def f_ab(x, a, b):
    return torch.where(x >= 0, a*x, b*torch.tanh(a*x/b))
```

| | |
|---|---|
|range|$(-b,\infty)$ — saturates to $-b$ below, unbounded above|
|smoothness|$C^2$, **not** $C^3$|
|at 0|$f(0)=0,\ f'(0)=a,\ f''(0)=0,\ f'''(0^-)=-2a^3/b^2,\ f'''(0^+)=0$|
|homogeneity|$f_{ka,kb}(x)=k f_{a,b}(x)$, and $f_{a,b}(x)=f_{1,b}(ax)$|

**Do not rewrite with ReLU.** `a*relu(x) - b*tanh(a*relu(-x)/b)` gives identical
values but **the gradient is wrong at exactly $x=0$** (PyTorch's `relu'(0)=0`
makes it 0; the correct value is $a$). `[verified]` — the discrepancy is exactly
$a$ at that one point. The `where` form is also 1.36x faster (0.178 ms vs
0.242 ms for 2M elements).

### 1.2 Decoder `g`

$$g_{A,B}(y)=\begin{cases}\dfrac{y}{A} & y\ge 0\\[6pt] \dfrac{B}{A}\,\mathrm{asinh}\!\big(\tfrac{y}{B}\big) & y<0\end{cases}\qquad A,B>0$$

```python
def g_AB(y, A, B):
    return torch.where(y >= 0, y/A, (B/A)*torch.asinh(y/B))
```

Defined on all of $\mathbb{R}$; no clamp needed. $g'(0)=1/A$ matches both sides.
$g'''(0^-)=-1/(AB^2)$. The negative side grows like $\ln(2u)$.

### 1.3 Why `asinh` and not `atanh`

$$\mathrm{atanh}\!\left(\frac{u}{\sqrt{1+u^2}}\right)=\mathrm{asinh}(u)$$

so the `asinh` form is "the `atanh` form with its argument smoothly squeezed
below 1". **Never write** `atanh(tanh(u))`: that is the identity on the reals, so
the whole thing collapses to $y/A$ and $B$ vanishes — and in float32,
$\tanh(10)$ rounds to 1.0 and `atanh(1.0) = inf`.

## 2. Where the nonlinearity comes from

### 2.1 `g` is not the inverse of `f`, and cannot be

The exact inverse of $g$ has $\big(g^{-1}\big)'''(0^-)=+A^3/B^2$, while
$f'''(0^-)=-2a^3/b^2$. Even with $a=A,\ b=B$ the signs differ:

$$f'''(0^-)-\big(g^{-1}\big)'''(0^-)=-\frac{3a^3}{b^2}\neq 0\quad\text{for all }a,b$$

`[verified]` This is the point. If $g$ *were* the exact inverse, consecutive
layers would telescope ($g\circ f=\mathrm{id}$) and every interior nonlinearity
would cancel, leaving a product of matrices.

### 2.2 The composition is cubic to leading order

$$f(x)=ax-\frac{a^3}{3b^2}x^3+\cdots,\qquad g(y)=\frac ya-\frac{y^3}{6ab^2}+\cdots$$

$$\boxed{\;g_{a,b}\big(f_{a,b}(x)\big)-x=-\frac{a^2}{2b^2}\,x^3+O(x^5)\qquad(x<0)\;}$$

`[verified]` at $a=2,b=0.5,x=-0.01$: measured $7.990\times10^{-6}$,
$a^2$ predicts $8.000\times10^{-6}$, $a^3$ predicts $1.600\times10^{-5}$.
**You need $a\neq1$ to pin the exponent.**

$$\boxed{\ \text{local nonlinearity}\ \propto\ (a/b)^2\ }$$

So $a$ is **not** a free gain that the incoming weights can absorb: moving it
changes the input scale relative to $b$, and $a/b$ is what sets the operating
point.

### 2.3 Two degeneracies to avoid

| degeneracy | cause | result |
|---|---|---|
|**telescoping**|$g$ exactly inverts $f$|interior nonlinearity cancels; a product of random matrices|
|**MLP collapse**|$g$ is linear|an ordinary MLP with activation $f$; $B$ stops mattering|

The `asinh` pair does neither.

## 3. The layer

$$\boxed{\;h^{(l+1)}_j = h^{(l)}_j + d_j\cdot f_{a_j,b_j}\Big(\sum_i c_{ij}\,g_{A_i,B_i}\big(h^{(l)}_i\big)\Big)\;}$$

$a,b,A,B$ are per-node reals kept positive by softplus. The residual is required
past depth ~24.

### 3.1 Normalising `C`

| scheme | property | cost |
|---|---|---|
|non-negative, $\sum_i c_{ij}=1$|quasi-arithmetic mean; **idempotent**; very depth-stable|**monotone map — hard accuracy ceiling**|
|**signed, $\sum_i\lvert c_{ij}\rvert=1$**|L1 gain bound $\lvert\sum_i c_{ij}z_i\rvert\le\max_i\lvert z_i\rvert$; non-expansive in sup norm; induces sparsity (~30% of edges die)|not idempotent; **does not stop decay** — hence the residual|
|ternary $\{0,\pm s\}$|$\lVert c\rVert_2$ depends only on the count|see 4|

**Signed L1 is the default.** Non-negative buys depth stability at the price of
monotonicity, which caps what the network can represent.

### 3.2 The gain `d`

| choice | when |
|---|---|
|learned per node|default; $n$ extra scalars|
|$1/\lVert c_{\cdot j}\rVert_2$, mean-normalised|**when `C` is sampled** — cancels the L2 jitter|
|**constant** $1/(s\sqrt n)$|**when `C` is binary/ternary** ($\lvert c\rvert$ constant $\Rightarrow\lVert c\rVert_2$ constant)|

`[verified]` **Do not compute $1/\lVert c\rVert_2$ for quantised `C`.** The
straight-through surrogate leaks into it and adds noise (MNIST depth 12:
0.9813 -> 0.9688).

### 3.3 Initialisation of the residual branch

$$d_j \leftarrow \frac{1}{\sqrt{L}}\qquad (L=\text{depth})$$

`[verified]` This matters more than any structural choice measured here.
Adding it to plain MLPs improved them 30-41% at depth 64, and SwiGLU **diverges
to NaN without it** even at depth 16. See RESULTS.md 3.

### 3.4 Separation of concerns

| part | what it prevents |
|---|---|
|$\sum_i\lvert c_{ij}\rvert=1$|**explosion** (L1 gain $\le1$)|
|$d_j=1/\lVert c\rVert_2$|**sampling jitter**|
|**residual + $1/\sqrt L$ init**|**decay** (without it, signal loses 21 orders of magnitude over 48 layers)|

## 4. Quantisation

### 4.1 Binary (BinaryConnect)

Keep a real *shadow* weight $\mu$; binarise **only in the forward pass**.

$$c_{ij}=s\cdot\mathrm{sign}(\mu_{ij}),\qquad s=1/\sqrt n$$

```python
def binary(mu, s):
    hard = torch.sign(mu)
    hard = torch.where(hard == 0, torch.ones_like(hard), hard)
    soft = torch.tanh(3.0 * mu)
    return s * (hard.detach() + soft - soft.detach())
```

`[verified]` **Do not sample `C` instead.** Drawing from a zero-mean
distribution does not train (accuracy 0.48). The sign must be *learned*.

### 4.2 Ternary

$$c_{ij}=s\cdot\mathrm{sign}(\mu_{ij})\cdot\mathbb{1}\big[\lvert\mu_{ij}\rvert>\theta\big]$$

Prefer ternary over binary: an L1-normalised layer lets ~30% of its edges fall
to zero on its own, and binary forces those back into service.

Effective bits per weight, with survival rate $p$:

$$\text{bits}=H(p)+p,\qquad H(p)=-p\log_2 p-(1-p)\log_2(1-p)$$

$p=0.5\Rightarrow1.5$ bits; $p=0.05\Rightarrow0.31$ bits. **Always report $p$.**
"1.58 bits" only holds at $p=1$.

### 4.3 Curvature-aware rounding `[untested]`

Thresholding on $\lvert\mu\rvert$ has no justification: it drops small-but-important
weights and keeps large-but-inert ones. Near an optimum the first-order term
vanishes, so the cost of rounding $w\to t$ is second-order:

$$\Delta L\approx\tfrac12 H_{ww}(t-w)^2$$

**Adam already holds a diagonal curvature estimate**: its second moment
$v_t\approx\mathbb{E}[g^2]$ is the Fisher / Gauss-Newton diagonal. So rank by

$$\boxed{\;\text{importance}_{ij}=\sqrt{v_{ij}}\cdot\lvert\mu_{ij}\rvert\;}$$

and keep the top $k$ per row (a fraction, not a threshold). The full treatment
(GPTQ / Optimal Brain Quantisation) builds $H=2XX^\top$ from the layer inputs
and compensates the remaining weights after each rounding:

$$\delta w_{-q}=-\frac{w_q}{[H^{-1}]_{qq}}[H^{-1}]_{-q,q}$$

### 4.4 Checks that are not optional

| check | why |
|---|---|
|**survival rate** ($c\neq0$)|**a layer of all zeros still passes signal through the residual**, so a dead network looks alive|
|ablation with $C\equiv0$|if the real network does not clearly beat it, the interior is doing nothing|
|effective bits $H(p)+p$|not 1.58 unless $p=1$|

## 5. Stochastic coupling

Jitter `C` during training only; **use the mean at inference** (a single
deterministic pass equals a 50-sample Monte-Carlo average).

$$c_{ij}=\mu_{ij}(1+\sigma\varepsilon_{ij}),\qquad\varepsilon\sim N(0,1)$$

- **multiplicative, not additive**
- $\sigma\approx0.1$; there is a cliff — at $\sigma=1.0$ the network falls back to
  its linear baseline
- **pair it with $d_j=1/\lVert c\rVert_2$**; alone it is a net loss (9x the variance)
- annealing $\sigma$ downward does not help: starting from a destructive level is
  not recoverable

## 6. Initialisation

| symbol | value |
|---|---|
|$a,A$|1.0|
|$b,B$|~0.1|
|$\mu$|$\mathcal{N}(0,0.5^2)$|
|$d$|$1/\sqrt L$ (3.3)|

### 6.1 Choosing `b` — measure, do not guess

`[verified]` **$b$ barely moves during training.** What matters is where you put it.

Measure the 99th percentile of $u=a\lvert x\rvert/b$ (for $x<0$), the argument
reaching `tanh`:

| 99th pct | state |
|---|---|
|$\ll1$|**saturation never engages**; $f$ is effectively linear. Lower $b$|
|$1$-$3$|good operating point|
|$\gg3$|fully saturated. Raise $b$|

Do the same for $\lvert y\rvert/B$ entering `asinh`.

## 7. Cheap inference

Train with exact `tanh`/`asinh`; substitute only at inference.

| substitution | speedup | cost (classification) |
|---|---|---|
|$\tanh\to\mathrm{clamp}(x,-1,1)$|**8.6x**|$\le0.002$|
|$\mathrm{asinh}\to x/\sqrt{1+x^2/3}$|**5.9x**|$\le0.002$|
|$\mathrm{asinh}\to$ identity|—|$\le0.002$|

`[verified]` **What is needed is the kink, not the curve.** hardtanh has a
maximum error of 0.238 over the range actually used and costs nothing; replacing
the nonlinearity with the identity costs 0.23 accuracy.

**Classification ceilings hide this.** The same substitutions measured on a
regression target showed quantisation costs of +27-45%. Check on a task without
a ceiling.

### 7.1 With $c\in\{\pm s\}$, $s$ a power of two

$c_{ij}x_i$ is a **1-bit shift and a sign flip**. $d_j$ is a constant. With
hardtanh the whole layer is shifts, adds and comparisons; the only multiplies
left are the three scalars $a,b,A$.

## 8. Diagnostics to run before trusting any number

**In this order. Do not proceed past a failure.**

| # | check | pass |
|---|---|---|
|1|**linear baseline** (Linear->Linear)|the model must beat it|
|2|**one-nonlinearity baseline** (Linear->$f$->Linear)|a deep model must clearly beat it|
|3|**depth value** = (2) / (deep)|**>= 1.5**; near 1 means the task does not need depth|
|4|99th pct into `tanh`|1-3 (6.1)|
|5|99th pct into `asinh`|same|
|6|nonlinearity $1-R^2$ of the model output vs its best linear fit|near 0 means the structure is idle|
|7|survival rate after quantisation|not 0 (4.4)|
|8|**ablation with $C\equiv0$**|the model must clearly beat it|
|9|seeds|>= 8 before claiming a difference; report difference / standard error|

Skipping 1, 2, 3 or 8 means reading a linear classifier's score as if it were a
nonlinear network's. We did exactly that twice (RESULTS.md 1).
