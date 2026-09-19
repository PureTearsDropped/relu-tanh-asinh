# Results

All numbers are normalised MSE (loss / target variance) on a teacher-student
task unless stated. Lower is better. Every table gives the seed count.

## 0. Setup

**Teacher** — a multiplicative interaction, chosen because it needs depth:

$$y=W_o\big(\tanh(W_Ax)\odot\tanh(W_Bx)\big),\quad x\sim N(0,1)^{32},\ W_{A},W_B\in\mathbb{R}^{128\times32},\ y\in\mathbb{R}^8$$

Data is generated fresh each step, so there is no overfitting and no ceiling.

**Baselines** (width 64, 5000 steps, Adam 3e-3 with cosine decay, 3 seeds):

| student | loss |
|---|---|
|Linear -> Linear (no nonlinearity)|0.91902 ± 0.00209|
|Linear -> $f$ -> Linear (no depth)|0.31051 ± 0.00430|
|4-layer residual MLP (gelu)|0.14628 ± 0.00131|

Depth value = 0.311 / 0.146 = **2.12**. A task scoring near 1 here is useless
(section 1).

## 1. A task that measured nothing

The first teacher was a random 4-layer GELU MLP with $N(0,1)$ inputs. Its
nonlinearity was $1-R^2 = 0.2227$, which looked adequate. It was not.

Raising the ternary threshold kept improving the loss. Counting the surviving
weights showed why:

| threshold | loss | fraction of $c\neq0$ |
|---|---|---|
|0.3|0.00141|0.539|
|1.0|0.00114|0.045|
|1.5|0.00109|0.0025|
|**2.5**|**0.00108 (best)**|**0.0000**|

**The best result had no surviving weights at all.** With $C=0$ the residual
passes $h$ through unchanged, so the four interior layers were inert; only
`Linear -> f -> Linear` was doing anything.

| student (4 seeds) | loss |
|---|---|
|**Linear -> Linear**|**0.00148**|
|**Linear -> $f$ -> Linear**|**0.00111**|
|4-layer residual MLP (gelu)|0.00146|

Every configuration measured on that task fell between those two numbers — relu
(0.00194) and PReMtan (0.00181) were **worse than linear**. A day of
comparisons was measuring noise. Withdrawn: scaling behaviour of binarisation,
per-activation quantisation costs, the conjugate layer's advantage, the optimal
$b_0$, channel and digit decompositions, threshold behaviour, "ternary beats
real".

**Signals that were visible and missed:** $a,b,A,B$ never moved from their
initial values; the 99th percentile into `tanh` was 0.23 (saturation never
engaged); the deep MLP scored the same as the linear student; ternarisation
*improved* accuracy.

### 1.1 Choosing a teacher

| teacher | $1-R^2$ | linear | one nonlinearity | deep | depth value |
|---|---|---|---|---|---|
|gelu, $N(0,1)$|0.2227|0.00148|0.00127|0.00177|**0.72**|
|gelu, $N(0,3)$|0.4301|0.03814|0.02906|0.03456|**0.84**|
|sin (random Fourier)|0.9983|0.99249|0.99252|0.99251|**1.00**|
|**product**|**0.9979**|**0.91782**|**0.30824**|**0.14574**|**2.12**|

High nonlinearity is not enough: the sin teacher scores 0.998 and no student
learns anything. Random deep networks do not use their depth — a same-width
shallow student matches them. **A task needs structure that only composition can
build.**

## 2. The conjugate layer against plain MLPs

Width 64, depth 4, 3 seeds, no residual-branch scaling:

| model | loss | vs linear |
|---|---|---|
|linear baseline|0.91902|1.00|
|one nonlinearity|0.31051|0.34|
|**MLP gelu**|**0.14628**|**0.16**|
|conjugate, real $C$, $b_0=0.1$|0.18005|0.20|
|MLP PReMtan|0.18432|0.20|
|conjugate, $b_0=0.03$|0.19263|0.21|
|conjugate, $b_0=0.5$|0.21695|0.24|

### 2.1 Quantisation

| model | real | ternary | cost | survival |
|---|---|---|---|---|
|MLP gelu|0.14628|0.21257|**+45%**|—|
|conjugate|0.18005|0.22801|**+27%**|0.546|
|conjugate (binary)|0.18005|0.25953|+44%|0.499|

The gap narrows under quantisation: gelu is 23% ahead with real weights and 7%
ahead with ternary.

## 3. Depth, and a second correction

With 16 seeds and **no** residual-branch scaling:

| depth | MLP gelu | conjugate | gap |
|---|---|---|---|
|4|0.14624 ± 0.00098|0.18068 ± 0.00156|+23.5%|
|8|0.14540 ± 0.00155|0.15847 ± 0.00234|+9.0%|
|16|**0.14243 ± 0.00170**|0.14515 ± 0.00110|+1.9%|
|**32**|0.15290 ± 0.00351|**0.13959 ± 0.00194**|**−8.7%**|

At depth 32 the ranges do not overlap across 16 seeds each (13 standard errors).
It looked like a structural result: worse when shallow, better when deep.

**It was an initialisation artefact.** SwiGLU needs its residual branch
initialised at $1/\sqrt{L}$ or it reaches NaN by depth 16, so that scaling had
been applied to SwiGLU alone. Applying it everywhere:

| depth 64, 8 seeds | without | **with** | change |
|---|---|---|---|
|relu|0.27164 ± 0.00494|**0.16099 ± 0.00506**|−41%|
|gelu|0.17293 ± 0.00747|**0.11773 ± 0.00081**|−32%|
|silu|0.17002 ± 0.00308|**0.11861 ± 0.00060**|−30%|
|SwiGLU|**NaN**|**0.05794 ± 0.00096**|—|

| depth 32 | without | with |
|---|---|---|
|gelu|0.15290|**0.11926**|
|conjugate|0.13959|**0.12216**|

**gelu gains 22% and overtakes.** The conjugate layer carries a learned per-node
gain $d_j$ and had been discovering the same $1/\sqrt{L}$ shrinkage on its own;
the plain MLP had no such freedom. The "different degradation curve" was the
presence of $d$, not the structure.

## 4. The comparison with everything matched

Depth 32, 8 seeds, residual branch at $1/\sqrt{L}$, matched parameter count:

| model | loss | std | parameters |
|---|---|---|---|
|**SwiGLU** ($m=n/3$)|**0.07095**|0.00086|133,704|
|MLP gelu|0.11926|0.00128|135,752|
|conjugate $f$-$g$|0.12216|0.00066|144,072|

Depth 64, 8 seeds:

| model | loss |
|---|---|
|SwiGLU, 3x parameters|**0.04949 ± 0.00070**|
|**SwiGLU**|**0.05794 ± 0.00096**|
|MLP gelu|0.11773 ± 0.00081|
|MLP silu|0.11861 ± 0.00060|
|MLP relu|0.16099 ± 0.00506|

**SwiGLU is 1.7x better than either alternative at depth 32 and 2x at depth 64.**
Nothing measured here shows the $f$-$g$ construction beating it.

## 5. Cheap inference

Train exact, substitute at inference. two-moons, width 32, depth 3, 8 seeds
(accuracy, higher is better):

| substitution | MLP with $f$ | conjugate |
|---|---|---|
|exact|0.9701|0.9687|
|$\tanh\to$ hardtanh|0.9681|0.9694|
|$\mathrm{asinh}\to$ identity|—|0.9719|
|both|—|0.9687|
|$\tanh\to$ identity|**0.7375**|—|

Throughput, 200k elements:

| | µs | speedup |
|---|---|---|
|tanh|42.9|1.0|
|**hardtanh**|**5.0**|**8.6x**|
|asinh|171.8|1.0|
|**$x/\sqrt{1+x^2/3}$**|**29.2**|**5.9x**|

hardtanh has a maximum error of 0.238 over the range actually used and costs
nothing measurable; removing the nonlinearity entirely costs 0.23. **The kink is
what matters, not the curve.**

MNIST, 10 classes, width 128, 4000 steps, 3 seeds:

| model | depth 3 exact | depth 3 cheap | depth 12 exact | depth 12 cheap | multiplier |
|---|---|---|---|---|---|
|MLP ($f$), real weights|**0.9826**|0.9825|**0.9835**|0.9833|yes|
|MLP ($f$), ±0.5 weights|0.9790|0.9792|0.9794|0.9788|no|
|conjugate, real $C$|0.9779|0.9778|0.9783|0.9763|yes|
|conjugate, ±0.5, $d=1/\lVert c\rVert_2$|0.9780|0.9764|0.9688|**0.9503**|no|
|**conjugate, ±0.5, $d$ constant**|0.9793|0.9792|**0.9813**|**0.9808**|no|

**Classification ceilings hide the cost.** The same substitutions on the
regression task (section 2.1) showed +27-45%. Do not conclude "free" from
accuracy alone.

## 6. Analytic results

$$f'''(0^-)=-\frac{2a^3}{b^2},\quad g'''(0^-)=-\frac{1}{AB^2},\quad\big(g^{-1}\big)'''(0^-)=+\frac{A^3}{B^2}$$

| $a$ | $b$ | $f'''$ measured | predicted | $g'''$ measured | predicted | difference | $-3a^3/b^2$ |
|---|---|---|---|---|---|---|---|
|1.0|0.5|−7.9988|−8.0000|−3.9993|−4.0000|−11.999|−12.000|
|1.0|0.1|−199.24|−200.00|−99.57|−100.00|−299.29|−300.00|
|2.0|0.5|−63.961|−64.000|−1.9997|−2.0000|−95.964|−96.000|

$$g_{a,b}\big(f_{a,b}(x)\big)-x=-\frac{a^2}{2b^2}x^3+O(x^5)$$

| $a$ | $b$ | $x$ | measured | $a^2$ | $a^3$ |
|---|---|---|---|---|---|
|1.0|0.5|−0.01|1.9994e−06|2.000e−06|2.000e−06|
|**2.0**|0.5|−0.01|**7.9904e−06**|**8.000e−06**|1.600e−05|
|**3.0**|0.5|−0.01|**1.7952e−05**|**1.800e−05**|5.400e−05|

The exponent is only identifiable with $a\neq1$; it was first written as $a^3$
and corrected after a reviewer pointed out the series expansion.

## 7. Open

- curvature-aware rounding (SPEC 4.3) — designed, never measured on a valid task
- why $a,b,A,B$ do not move during training
- channel and digit decompositions of the ternary coupling — the measurements
  that existed were on the invalid task
- whether the conjugate layer helps anywhere SwiGLU does not
