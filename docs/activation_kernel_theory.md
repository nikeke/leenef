# The role of the nonlinearity in leenef

*Companion note to SSUP-I §2 and §5.4.*  This note answers a general
question: **what does the pointwise activation actually do in the leenef
analytic solve, and which activations are equivalent, weaker, or
stronger than `abs`?**  The headline is that leenef is explicit
random-feature ridge regression, so *the activation is a choice of
kernel*.  Everything else — the linear-collapse argument, the
`abs`/ReLU equivalence, the weakness of `tanh`/`step`, and the
existence of strictly richer choices — follows from one object, the
**dual activation kernel**.

---

## 1. Setup

A leenef layer maps $x \in \mathbb{R}^{d_{\mathrm{in}}}$ to activities

$$
a_i = G(z_i), \qquad z_i = g_i\, e_i^\top x + b_i, \quad i = 1,\dots,n,
$$

with fixed encoders $e_i$, gains $g_i$, biases $b_i$, and a pointwise
activation $G$.  Only the decoder $D \in \mathbb{R}^{n\times d_{\mathrm{out}}}$ is
fit, by ridge regression:

$$
D = (A^\top A + \alpha_{\mathrm{eff}} I)^{-1} A^\top Y .
$$

The decoded predictor is $f_D(x) = \sum_i D_i\, G(z_i)$.  What the solve
can represent is exactly the span of the feature columns
$\{x \mapsto G(z_i)\}_{i=1}^n$ (shrunk by $\alpha_{\mathrm{eff}}$).  So the relevant
object is the **hypothesis class**

$$
\mathcal{F}_G = \Big\{\, x \mapsto \textstyle\sum_i D_i\, G(g_i e_i^\top x + b_i) \;:\; D \in \mathbb{R}^{n\times d_{\mathrm{out}}} \,\Big\}.
$$

Two activations that induce the same $\mathcal{F}_G$ (up to a bounded,
$n$-independent correction) are indistinguishable to the analytic solve.

---

## 2. leenef is random-feature ridge regression; the activation is a kernel

Draw $(e_i, b_i)$ i.i.d. from the encoder/bias prior $P$.  Then the
empirical kernel

$$
\frac{1}{n}\, (A A^\top)_{jk} = \frac{1}{n}\sum_i G(z_i^{(j)})\,G(z_i^{(k)})
\;\xrightarrow[n\to\infty]{}\;
k_G(x_j, x_k) := \mathbb{E}_{(e,b)\sim P}\big[\, G(g e^\top x_j + b)\, G(g e^\top x_k + b)\,\big]
$$

by the law of large numbers (this is the random-features → kernel limit
of Rahimi & Recht, 2007; the network-as-kernel limit of Neal, 1996 and
Williams, 1997).  Ridge regression on the explicit features converges to
**kernel ridge regression with kernel $k_G$**.  Hence:

> **The activation $G$ selects the kernel $k_G$.**  Choosing an activation
> *is* choosing the function space the analytic solve searches.  Width
> $n$ only controls how well the explicit features approximate $k_G$.

This single statement organizes everything below.

*Two caveats on the limit.*  (i) For the explicit-features solve to
converge to kernel ridge with a **fixed** regularization $\lambda$, the
ridge penalty must track the feature count: with the normalization above
($\tfrac1n AA^\top \to k_G$) one takes $\alpha_{\mathrm{eff}} = n\lambda$,
or equivalently divides the features by $\sqrt n$.  leenef's trace-scaled
$\alpha$ (SSUP-I §4) supplies exactly this kind of scale-tracking penalty,
which is why the kernel picture is the right asymptotic model of the
*regularized* solve and not just the interpolating one.  (ii) The limit
object $k_G$ depends on the encoder/bias prior $P$; the clean Hermite
spectrum below is for the **Gaussian** instance of $P$.

### 2.1 The dual kernel is a power series in the cosine (Mehler)

Take the zero-bias, isotropic case for a clean spectral picture: $b=0$,
$e$ uniform on the sphere, and normalize so that for unit inputs
$x,y$ the pair $(e^\top x, e^\top y)$ is bivariate standard normal with
correlation $\rho = x^\top y = \cos\theta$.  Expand $G$ in the
probabilists' Hermite basis $\mathrm{He}_k$ (orthogonal under the
Gaussian weight, $\mathbb{E}[\mathrm{He}_j\mathrm{He}_k]=k!\,\delta_{jk}$):

$$
G(t) = \sum_{k\ge 0} \frac{a_k}{k!}\,\mathrm{He}_k(t), \qquad a_k = \mathbb{E}\big[G(Z)\,\mathrm{He}_k(Z)\big],\; Z\sim\mathcal{N}(0,1).
$$

Mehler's formula then gives the dual kernel **exactly** as

$$
\boxed{\,k_G(\rho) = \mathbb{E}[G(U)G(V)] = \sum_{k\ge 0} \frac{a_k^2}{k!}\,\rho^{\,k}\,}
\qquad (U,V)\ \text{std. normal, corr.}\ \rho .
$$

The kernel's entire content is the **squared Hermite spectrum**
$\{a_k^2\}$ of the activation.  Read off the consequences:

- $a_0$ (the mean, $\mathrm{He}_0=1$) contributes a constant $a_0^2$: a
  rank-one DC term, removable by the bias/intercept feature.
- $a_1$ (the linear mode, $\mathrm{He}_1=t$) contributes $a_1^2\,\rho$: a
  **pure linear kernel**.  By itself it can only ever represent a linear
  map $x\mapsto Wx$ — no matter how large $n$ is.
- $a_k,\ k\ge 2$ contribute $\rho^k$: these are the genuinely nonlinear
  pieces of the kernel.  Each nonzero $a_k$ adds a new family of
  degree-$k$ harmonics; an activation with infinitely many nonzero
  $a_k$ (e.g. `relu`, `gauss`) induces an infinite-dimensional — hence
  universal — function space, whereas one with finitely many (e.g.
  `square`) induces a finite-dimensional one.

> **Discriminative power lives in Hermite modes $k\ge 2$.**  The constant
> mode is the bias feature; the linear mode is the *collapse mode*; only
> $k\ge 2$ buys anything a linear readout on raw $x$ could not.

*Exact vs. approximate.*  Mehler's formula is **exact** for Gaussian
encoders (each $z_i$ then jointly Gaussian).  leenef's default
`hypersphere` encoders put $e_i$ on the unit sphere, for which the exact
dual kernel is the **Gegenbauer/spherical-harmonic** series on
$S^{d_{\mathrm{in}}-1}$; in high dimension this converges to the Hermite
series above (a single spherical marginal is asymptotically Gaussian), so
we use the Hermite spectrum as the clean spectral proxy and validate it
directly on Gaussian data in §7.

---

## 3. The linear-collapse theorem, recovered

If $G$ is affine, $G(t)=c_0+c_1 t$, then $a_k=0$ for all $k\ge 2$, so
$k_G(\rho)=c_0^2 + c_1^2\rho$ — the linear (dot-product-plus-constant)
kernel.  Kernel ridge with a linear kernel is ordinary ridge on $x$:
the predictor is $f(x)=Wx+c$ regardless of $n$.  This is exactly the
SSUP-I §2 collapse, now stated as "the linear activation selects the
linear kernel."  The minimal repair is therefore precise:

> **A useful activation is one with nonzero Hermite energy in some mode
> $k\ge 2$.**  Anything else over-parameterizes a linear map.

---

## 4. Single-kink activations are all `abs` (exact decomposition; equivalent up to a bounded affine term)

The pointwise *decomposition* below is exact and needs **no** Gaussian or
large-$n$ assumption; the resulting hypothesis-space equivalence holds up
to a bounded, $n$-independent affine correction (and becomes an exact
equality of closures only in the universal $n\to\infty$ limit).  A
*bent-line* activation is any continuous piecewise-linear $G$ with a
single kink at the origin and slopes $m_+$ (right) and $m_-$ (left).
Write

$$
G(t) = c_0 + c_1 t + c_2 |t|, \qquad c_1 = \tfrac{m_+ + m_-}{2}, \quad c_2 = \tfrac{m_+ - m_-}{2}.
$$

Then for **every** neuron, pointwise and for any fixed $b_i$,

$$
a_i = G(z_i) = c_0 + c_1\,(g_i e_i^\top x + b_i) + c_2\,|z_i|,
$$

so the decoded function splits as

$$
f_D(x) = \underbrace{c_1 \sum_i D_i\,(g_i e_i^\top x + b_i)}_{\text{affine in } x,\ \dim \le d_{\mathrm{in}}+1} \;+\; \underbrace{c_2 \sum_i D_i\,|z_i|}_{=\,c_2\,\cdot\,\text{the } \texttt{abs}\text{ decode}} .
$$

Hence

$$
\mathcal{F}_{\text{bent}} \subseteq \mathrm{Affine}(x) + \mathcal{F}_{\texttt{abs}},
$$

and because `abs` features with biases are universal (they already
contain affine maps in their closure), the *closures* of
$\mathcal{F}_{\text{bent}}$ and $\mathcal{F}_{\texttt{abs}}$ coincide in
the $n\to\infty$ limit.  At finite $n$ the two are not literally the same
class — the **single shared decoder $D$** must simultaneously serve the
$c_2\,|z|$ part and the $c_1$ affine part — but they differ only by a
cheap $(d_{\mathrm{in}}+1)$-dimensional affine slack that does not grow
with $n$.  Spectrally (§2.1) this is the statement that

$$
a_k(\text{bent}) = c_2\, a_k(\texttt{abs}) \quad (k\ge 2), \qquad
a_1(\text{bent}) = c_1 + c_2\,a_1(\texttt{abs}),
$$

so $k_{\text{bent}}(\rho) - c_2^2\,k_{\texttt{abs}}(\rho)$ is a degree-$\le 1$
polynomial in $\rho$: the two kernels share their entire nonlinear part
up to a constant factor and a linear term.

**Corollaries.**

- **ReLU $=$ `abs` up to linear.**  $\mathrm{ReLU}(t)=\tfrac12 t+\tfrac12|t|$,
  i.e. $m_+=1,m_-=0$, $c_1=c_2=\tfrac12$.  Its nonlinear kernel is
  $\tfrac14$ of `abs`'s plus a linear $\tfrac14\rho$.  Same hypothesis
  space.  (One `abs` neuron is informationally a *mirrored pair* of ReLU
  neurons, as SSUP-I §5.4 notes.)
- **The requested monotone kink.**  Slopes $(1,\tfrac12)$ —
  $G(t)=t$ for $t\ge0$, $\tfrac12 t$ otherwise, equivalently
  $G(t)=\tfrac34 t + \tfrac14|t|$ — is a *monotone* relative of `abs`
  that merely changes the angle at the origin.  It carries
  $c_2=\tfrac14$ of `abs`'s nonlinear energy.  Prediction: **ties `abs`/
  ReLU** to within the bounded affine slack.
- **The slope is irrelevant (as long as there is a kink).**  Any two
  positive slopes work; the nonlinear energy is $((m_+-m_-)/2)^2$.  As
  $m_-\to m_+$ the kink vanishes, $c_2\to 0$, and the activation
  collapses to linear — recovering §3 continuously.  Leaky-ReLU
  (slope $0.1$) and Softplus (a smoothed kink) are further members and
  are predicted equivalent to ReLU.

### 4.1 The origin angle: one knob through the whole kink family

A vivid way to see "the slope is irrelevant, the kink is everything" is to
parameterize the bent line by the **interior angle $\theta$ at the
origin**.  Fix the right arm to slope $1$ (a $45^\circ$ ray) and let the
left arm have slope $b=\tan(\tfrac\pi4+\theta)$, so $\theta$ is literally
the angle between the two arms.  Then

$$
\theta=\tfrac\pi2 \to \texttt{abs},\qquad
\theta=\tfrac{3\pi}4 \to \texttt{ReLU},\qquad
\theta=\pi \to \text{identity (linear collapse)},
$$

with the valid range $\theta\in(\tfrac\pi4,\tfrac{5\pi}4)$.  In the
$G=c_1 t + c_2|t|$ coordinates, $c_1=\tfrac{1+b}2$, $c_2=\tfrac{1-b}2$,
so the nonlinear energy is $c_2^2 = \big(\tfrac{1-b}2\big)^2$, which
vanishes **only** at $b=1$, i.e. $\theta=\pi$.  Every angle reuses the
*same* features $z$ and $|z|$; only the scalar mix $(c_1,c_2)$ changes.

> **Prediction.**  Sweeping $\theta$ from $\tfrac\pi4$ to $\tfrac{5\pi}4$,
> test accuracy is **flat** (all angles share `abs`'s $k\ge2$ spectrum)
> except for a **collapse dip at exactly $\theta=\pi$**.  Because the
> trace-scaled $\alpha$ shrinks a vanishingly small $c_2$, the dip has a
> finite width set by $\alpha/n$ rather than being a measure-zero spike:
> accuracy degrades smoothly only in a narrow neighborhood of $\pi$ where
> $c_2^2$ falls below the regularization floor.

This is verified in §8.2.

---

## 5. Why `tanh` and `step` are weaker

Both discard magnitude, which is where the even ($k\ge2$) Hermite energy
of `abs` lives.

- **`step`** (our $G=\mathbf{1}[t>0]$, the $0/1$ Heaviside) has
  $a_0=\tfrac12$, $a_1=\tfrac{1}{\sqrt{2\pi}}$, and — because
  $\Theta-\tfrac12$ is *odd* — all **even** modes $a_2=a_4=\dots=0$.
  Its dual kernel is the order-0 arc-cosine kernel
  $\mathbb{E}[\Theta(u)\Theta(v)] = \tfrac12 - \tfrac{\theta}{2\pi}$
  (Cho & Saul, 2009; the $\pm1$ sign version gives $1-\theta/\pi$), a
  correlation-only (magnitude-blind) kernel.  Weakest in the panel.
- **`tanh`** saturates.  At small gain it is nearly linear (energy
  concentrated in $a_1$ → near-collapse); at large gain it approaches
  $\mathrm{sign}$ → `step`-like.  Either way it suppresses the magnitude
  modes, so it trails `abs`/ReLU at every width.

This is the frame-coefficient reading of SSUP-I §2 made precise: a good
activation must **preserve coefficient magnitude**, i.e. carry even
Hermite energy, which `abs` does and `tanh`/`step` do not.

---

## 6. Going beyond `abs`: changing the $k\ge2$ spectrum

To *beat* `abs` you must change the nonlinear spectrum, not just re-mix
the constant/linear modes.  Candidates, with their kernels:

- **Power family $|t|^p$.**  Same sign-fold as `abs` ($p=1$) but a
  different even spectrum.
  - $p=2$ (**square**): $a_0=1$, $a_2=2$, all else $0$ ⇒
    $k(\rho)=1+2\rho^2$, the **degree-2 polynomial kernel**.  Rich but
    *finite-dimensional* (degree-$\le2$ monomials, $O(d_{\mathrm{in}}^2)$
    features) — explains why square is competitive yet not a robust
    universal win.
  - $p=\tfrac12$ (**`sqrt_abs`**): heavier low-frequency (slowly decaying)
    even spectrum — a different smoothness prior.
- **Multi-fold $\big||t|-c\big|$ (`double_fold`).**  Two kinks per
  neuron ⇒ richer even spectrum than a single `abs` fold at the same
  width.  A cheap way to add nonlinear modes per neuron.
- **`cos` (random Fourier feature).**  With Gaussian-like encoders and
  gain setting the bandwidth, a population of $\cos(g e^\top x + b)$
  features approximates a **stationary (RBF/Gaussian) kernel**
  (Rahimi & Recht, 2007): $k(x,y)\approx \exp(-\sigma^2\|x-y\|^2/2)$ with
  $\sigma\propto$ gain.  Universal in principle — but two caveats matter
  in leenef.  *(a) Bandwidth (gain) sensitivity:* too small a gain leaves
  $\cos$ nearly constant (collapse), too large a gain decorrelates every
  pair (white kernel).  *(b) leenef is not the textbook RFF construction:*
  the clean Rahimi–Recht estimator uses Gaussian weights with a **random
  phase** $b\sim U(0,2\pi)$ and no data term; leenef's data-driven bias
  makes $z = g\,e^\top(x-c)$, a *centered* feature, not a random-phase
  one.  A revealing consequence of the small-gain limit: since
  $\cos(z)\approx 1-\tfrac12 z^2$, **`cos` at small gain is a `square`
  feature in disguise** (a degree-2 polynomial kernel), not a collapse —
  so a moderate-gain `cos` that still scores well is exploiting its
  quadratic ($a_2$) energy, not yet the full RBF.  The gain sweep in the
  panel separates these regimes.
- **`gauss` bump $\exp(-t^2/2)$.**  With data-driven biases each neuron
  is a soft RBF *receptive field* centered on a training sample along
  $e_i$.  Also an RBF-family kernel; bandwidth set by gain.

**The bandwidth point is the whole game for the RBF family.**  In leenef
the gain $g_i$ is exactly the kernel bandwidth: $z_i = g_i e_i^\top x + b_i$,
so the per-neuron gain spread $U(g_{\mathrm{lo}},g_{\mathrm{hi}})$ is a
multiscale RBF — a built-in bandwidth mixture.  This is why the gain
range matters far more for `cos`/`gauss` than for the kink family
(where SSUP-I §5.4 found gain nearly irrelevant): for a kink, gain only
rescales $a_k\!\to\! g^? a_k$ uniformly and trace-scaled $\alpha$ undoes
it; for an RBF feature, gain *moves energy between Hermite modes* and so
reshapes the kernel.

---

## 7. Predictions (tested in the `c2_altact` panel)

| activation | kernel / spectrum | prediction vs `abs` |
|---|---|---|
| `linear` | linear kernel ($a_1$ only) | collapse (SSUP-I §2) |
| `abs` | even arc-cosine, $a_0,a_2,a_4,\dots$ | reference |
| `relu` | $\tfrac14\times$`abs` nonlinear $+$ linear | tie |
| `bent_line` (1, ½) | $\tfrac{1}{16}\times$`abs` nonlinear $+$ linear | **tie** |
| `leaky_relu` (0.1) | bent-line member | tie |
| `softplus` | smoothed kink | tie |
| `tanh` | saturating, magnitude-suppressing | worse |
| `step` | order-0 arc-cosine (magnitude-blind) | worse |
| `square` | degree-2 polynomial kernel | $\approx$ / mild CIFAR edge |
| `sqrt_abs` | heavier low-freq even spectrum | $\approx$ or slightly worse |
| `double_fold` | richer even spectrum, 2 folds/neuron | $\approx$ to slightly better |
| `cos` | RBF / random Fourier | **gain-dependent**; can match or beat |
| `gauss` | RBF receptive field | **gain-dependent**; can match or beat |

The empirical panel that tests these predictions lives at
`results/sl_clean/c2_altact_{mnist,fashion,cifar10}_panel.json`
(10 seeds, $n=5000$, hypersphere encoders, data-driven biases,
trace-scaled $\alpha=10^{-3}$, MNIST / Fashion-MNIST / CIFAR-10), with
gain sweeps for `cos` and `gauss`.

---

## 8. Empirical validation

### 8.1 Direct test of the spectrum: Hermite-mode recovery

The cleanest test of §2–§3 is to ask the solve to learn a **single
Hermite mode** and check that it succeeds exactly when that mode is in the
activation's spectrum.  On isotropic Gaussian data
$x\sim\mathcal N(0,I_{20})$ with Gaussian unit encoders, zero bias and
unit gain (the regime where Mehler is exact), we fit ridge on $n=2000$
random features to each pure-mode target $y_k(x)=\widehat{\mathrm{He}}_k(\langle e_0,x\rangle)$,
$k=1,2,3$, and report held-out $R^2$ (mean over 5 seeds).  The runner is
`benchmarks/sl_clean/hermite_mode_recovery.py`.

| activation | $a_1^2$ | $a_2^2$ | $a_3^2$ | $R^2_{k=1}$ | $R^2_{k=2}$ | $R^2_{k=3}$ |
|---|---|---|---|---|---|---|
| `linear` | 1.00 | 0 | 0 | **1.00** | $\le0$ | $\le0$ |
| `abs` | 0 | 0.32 | 0 | $\le0$ | **0.83** | $\le0$ |
| `relu` | 0.25 | 0.08 | 0 | **1.00** | **0.83** | $\le0$ |
| `bent_line` (1,½) | 0.56 | 0.02 | 0 | **1.00** | **0.84** | $\le0$ |
| `square` | 0 | 2.00 | 0 | $\le0$ | **0.95** | $\le0$ |
| `tanh` | 0.37 | 0 | 0.02 | **0.98** | $\le0$ | **0.20** |
| `cos` | 0 | 0.18 | 0 | $\le0$ | **0.82** | $\le0$ |
| `gauss` | 0 | 0.06 | 0 | $\le0$ | **0.76** | $\le0$ |

The pattern is exactly the theory: **$R^2_k>0$ iff $a_k^2>0$.**  In
particular —

- **`linear` learns only $k=1$ and nothing nonlinear** — the linear
  collapse of §3, shown directly.
- **`abs`/`relu`/`bent_line` all recover $k=2$ with the same $R^2$**:
  the kink family shares the even spectrum (§4), so it ties on any target
  whose signal sits in low even modes — which is what real
  image-classification targets largely are.
- **`relu` cannot learn the cubic $k=3$** ($a_3=0$ exactly, since
  $\mathrm{ReLU}=\tfrac12(t+|t|)$ inherits `abs`'s vanishing odd modes),
  whereas **`tanh` can** — the lone tested activation with odd
  $k\ge3$ energy.  A genuinely counterintuitive, confirmed prediction.
- **`square` is pure degree-2**: it recovers $k=2$ (most strongly, its
  $a_2^2$ is largest) and nothing else.
- **`cos`/`gauss` are even-only** here (zero bias makes them even
  functions): they recover $k=2$ but not the odd modes.

($R^2\le0$ denotes "cannot represent": the regularized fit does no better
than the mean.  The quartic $k=4$ mode is not reliably learned by anyone
at this $(n, N_{\text{train}})$ because its $a_4^2$ is small — a
finite-sample effect consistent with learnability $\propto a_k^2$.)

### 8.2 The origin-angle sweep

Sweeping the interior angle $\theta$ (§4.1) on the SSUP-I §5.4 protocol
($n=5000$, hypersphere encoders, data-driven biases, trace-scaled
$\alpha=10^{-3}$, **10 seeds**) confirms the prediction precisely on all
three datasets.  Test accuracy (mean over 10 seeds; SE $\le 0.001$
throughout except as noted):

| $\theta/\pi$ | 0.50 `abs` | 0.75 `relu` | 0.90 | 0.95 | 0.98 | 0.99 | **1.00** | 1.01 | 1.02 | 1.05 | 1.10 | 1.23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | 0.9704 | 0.9705 | 0.9685 | 0.9642 | 0.9512 | 0.9315 | **0.8603** | 0.9316 | 0.9513 | 0.9641 | 0.9683 | 0.9701 |
| Fashion | 0.8778 | 0.8765 | 0.8727 | 0.8658 | 0.8520 | 0.8363 | **0.8112** | 0.8363 | 0.8516 | 0.8660 | 0.8728 | 0.8770 |
| CIFAR-10 | 0.4945 | 0.4945 | 0.4935 | 0.4874 | 0.4638 | 0.4409 | **0.3880** | 0.4403 | 0.4641 | 0.4870 | 0.4919 | 0.4938 |

Three signatures, all predicted:

1. **A flat plateau** at the dataset's `abs`-level accuracy across the
   entire arm-angle range $\theta/\pi\in[0.27,1.23]$ — every angle shares
   `abs`'s $k\ge2$ spectrum, so they are statistically indistinguishable
   (the largest plateau spread, $\sim0.2\%$ on MNIST, is within $\sim2$ SE).
2. **A collapse dip at exactly $\theta=\pi$** of the right magnitude
   (down to the *linear-readout* accuracy of each dataset), recovering
   smoothly within a narrow $\pm0.03\pi$ basin whose finite width is set
   by the $\alpha/n$ regularization floor — not a measure-zero spike.
   The basin is **symmetric** ($\theta/\pi=0.99$ and $1.01$ agree to
   $10^{-4}$, etc.), because the two arms induce the same $|z|$ spectrum.
3. **Zero seed variance at $\theta=\pi$** (SE $=0.0000$): at the linear
   angle the features are exactly $z=g\,e^\top(x-c)$, so the readout is a
   linear classifier whose accuracy is *independent of the random
   encoders* — every seed collapses to the identical number.  This is a
   crisp, unmistakable fingerprint of the §3 collapse.

The sweep at $\theta=\tfrac\pi2$ reproduces the published `abs` cell
($0.9704$ MNIST) and at $\theta=\tfrac{3\pi}4$ the `relu` cell to within
scale-invariance, validating the shared-feature recombination used to
compute it.  Full panels:
`results/sl_clean/kink_angle_{mnist,fashion,cifar10}_panel.json`.

### 8.3 The real-task activation panel

The full-dataset comparison of all activations (10 seeds, $n=5000$,
hypersphere encoders, data-driven biases, trace-scaled $\alpha=10^{-3}$,
MNIST / Fashion-MNIST / CIFAR-10), with gain sweeps for `cos`/`gauss`,
lives at `results/sl_clean/c2_altact_{mnist,fashion,cifar10}_panel.json`.
It tests the §7 predictions on real targets; the kink family is expected
to tie `abs`, `tanh`/`step` to trail, and `square`/`cos`/`gauss` to be
competitive (gain-dependent for the RBF pair).

---

## 9. Summary

1. **leenef = kernel ridge regression**; the activation chooses the
   kernel $k_G(x,y)=\mathbb{E}_{e,b}[G(z)G(z')]$, whose content is the
   squared Hermite spectrum $\{a_k^2\}$ of $G$.
2. **The linear mode ($k=1$) is the collapse mode**; all discrimination
   lives in $k\ge2$.  Linear activation = linear kernel = §2 collapse.
3. **Every single-kink (bent-line) activation equals `abs` up to a
   bounded affine correction** — exactly, for any bias.  ReLU,
   leaky-ReLU, the requested monotone $(1,\tfrac12)$ kink, and Softplus
   are all `abs` to the solve.  This *explains* the SSUP-I §5.4
   `abs`$\approx$ReLU result and predicts the new kink ties.
4. **`tanh`/`step` are weaker** because they discard the magnitude (even
   Hermite) modes that carry discrimination.
5. **To beat `abs` you must change the $k\ge2$ spectrum**: `square`
   (degree-2 polynomial kernel), `double_fold` (more folds), or the RBF
   family (`cos`, `gauss`) whose **bandwidth is the per-neuron gain**.
   These are the principled "simpler, more natural, or more powerful"
   alternatives.

### References

- Rahimi & Recht (2007), *Random Features for Large-Scale Kernel
  Machines*.
- Cho & Saul (2009), *Kernel Methods for Deep Learning* (arc-cosine
  kernels).
- Neal (1996); Williams (1997) — infinite networks as Gaussian-process
  kernels.
- Eliasmith & Anderson (2003) — the Neural Engineering Framework.
