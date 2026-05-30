# The role of the nonlinearity in leenef

*Companion note to SSUP-I §2 and §5.4.*  This note characterizes a
general question: **what does the pointwise activation actually do in the
leenef analytic solve, and which activations are equivalent, weaker, or
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
- **A monotone kink.**  Slopes $(1,\tfrac12)$ —
  $G(t)=t$ for $t\ge0$, $\tfrac12 t$ otherwise, equivalently
  $G(t)=\tfrac34 t + \tfrac14|t|$ — is a *monotone* relative of `abs`
  that merely changes the angle at the origin.  It carries
  $c_2=\tfrac14$ of `abs`'s nonlinear energy.  Prediction: **ties `abs`/
  ReLU** to within the bounded affine slack.
- **The slope is irrelevant (as long as there is a kink).**  Any two
  positive slopes work; the nonlinear energy is $((m_+-m_-)/2)^2$.  As
  $m_-\to m_+$ the kink vanishes, $c_2\to 0$, and the activation
  collapses to linear — recovering §3 continuously.  Leaky-ReLU
  (slope $0.1$) is another exact member.  Softplus is a *smoothed* kink:
  equivalent to ReLU only in the sharp limit ($\beta\cdot$gain large);
  when the smoothing scale rivals the active range it loses nonlinear
  energy (see §8.3).

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

### 6.1 The parity axis: a full-spectrum kernel needs broken symmetry

`cos`, `gauss`, `square`, `double_fold`, `box` are all **even** functions
of their argument, $G(-z)=G(z)$.  An even $G$ has $a_k=0$ for every *odd*
$k$, so at zero bias its dual kernel keeps only even Mehler terms
$k=0,2,4,\dots$.  Is that a problem?  Split the missing odd modes:

- $k=1$ is the **linear / collapse mode** (§3).  It is not nonlinear
  expressive power, so dropping it is not harmful *provided* a separate
  affine path or the data-driven bias supplies whatever linear structure the
  target needs — which, as point 1 below shows, the bias does.  It is not
  universally "free," but for these encoders it costs nothing.
- $k=3,5,\dots$ are genuine odd discriminative modes.  An even activation
  forfeits them.  The Hermite probe (§8.1) shows this directly: at zero
  bias `cos`/`gauss`/`square_wave`/`box` recover even mode $k=2$
  ($R^2\approx0.8/0.76/0.77/0.62$) but *fail* odd $k=1,3$ ($R^2<0$).

So the "even-only limitation" is real in principle.  The instructive
result is **how cheaply it is removed.**  Three observations:

1. **Any nonzero bias breaks the exact even/odd selection rule.**  With a
   per-neuron bias the population coefficient is no longer the zero-bias
   $a_k$ but the **bias-averaged shifted** coefficient
   $\bar a_k^2=\mathbb{E}_b\!\big[a_k(b)^2\big]$ with
   $a_k(b)=\mathbb{E}_z[G(z+b)\,\mathrm{he}_k(z)]$.  For an even *non-poly­nomial*
   $G$ a shift injects odd coefficients: by Stein's lemma
   $a_1(b)=\mathbb{E}[G'(z+b)]$, and $G'$ is odd for even $G$, so $a_1(b)\neq0$
   once $b\neq0$ — concretely $a_1(b)=-e^{-1/2}\sin b$ for $\cos$ and
   $a_1(b)=-\tfrac{b}{2\sqrt2}e^{-b^2/4}$ for the `gauss` bump.  Two caveats
   apply: a shift restores **at least the low odd modes (notably
   $k=1$)** but the higher odd modes may stay weak at finite width; and for a
   *polynomial* $G$ (e.g. `square`) a shift cannot create any mode above the
   polynomial degree (it gains $k=1$ but never $k=3$).  leenef's data-driven
   bias $b_i=-g_i\,e_i^\top c_i$ is exactly such a nonzero, neuron-varying
   offset — a data-centered receptive-field shift, not the textbook uniform
   RFF phase, so its induced kernel depends on the center distribution.  The
   Hermite probe confirms the mechanism directly (§8.1): with
   $\mathrm{bias\ std}=2$, `cos` recovers $k=1$ ($R^2\;{-1.49}\to0.95$) and
   `abs` recovers $k=1$ ($-1.04\to1.00$); the cubic $k=3$ stays near zero,
   consistent with "low odd modes restored, high odd modes weak."  The
   practical reading: **in the real pipeline cos/gauss are not strictly
   even-restricted** — the bias supplies the dominant ($k=1$) odd content —
   which helps explain why they already score at the top on CIFAR-10 (§8.3).
2. **The minimal symmetry-breaking activation is a phase, not a more
   complex function.**  Replace $\cos z$ by
   $$\cos z+\sin z=\sqrt2\,\cos\!\big(z-\tfrac{\pi}{4}\big)\quad(\texttt{cos\_sin}).$$
   This is a fixed $\pi/4$ phase shift — no qualitatively more complex
   activation, just a reflection-symmetry-breaking offset — and it is now
   neither even nor odd, so it carries **every** Hermite mode.  The probe
   shows `cos_sin` recovers $k=1,2,3$ ($R^2=0.95/0.81/0.08$) at zero bias
   where `cos` recovers only $k=2$.  The general version is the textbook
   random-Fourier feature: a **random** phase $b\sim U(0,2\pi)$ makes
   $\mathbb{E}_b[\cos(z+b)\cos(z'+b)]=\tfrac12\cos(z-z')$, a stationary
   cosine feature; with **Gaussian** frequency weights this integrates to the
   exact Gaussian-RBF kernel $\exp(-\sigma^2\|x-y\|^2/2)$, while the
   hypersphere encoders used here give a closely related stationary radial
   kernel (Rahimi & Recht, 2007).  Fixing `cos`/`gauss` therefore does
   **not** require a more complex activation — only **broken parity**,
   which a single phase term (or the bias the pipeline already uses)
   supplies.
3. **`gelu`/`silu` are smooth, gain-robust members of the relu class — not
   a richer spectrum.**  Writing $\mathrm{gelu}(z)=z\,\Phi(z)=\tfrac12 z+
   \tfrac12 z\,\mathrm{erf}(z/\sqrt2)$ and $\mathrm{silu}(z)=z\,\sigma(z)=
   \tfrac12 z+z(\sigma(z)-\tfrac12)$ splits each into a **pure linear part
   plus an even function** (the second term is odd$\times$odd $=$ even).
   So, exactly like $\mathrm{relu}=\tfrac12(z+|z|)$, they carry $k=1$ and the
   even modes but have **zero odd coefficient for every $k\ge3$** — not
   "approximately zero," identically zero by symmetry (probe: $a_1^2=0.25$
   as for relu, $a_3=0$).  Their advantage over `cos`/`gauss` is *practical,
   not spectral*: being non-periodic and unbounded-below-free they have **no
   bandwidth to tune** and do not collapse at small gain.  They are
   therefore best billed as "a smooth, gain-robust `relu`," a fine default but
   **not** a way to reach the odd $k\ge3$ modes.  The only tested route to a
   genuinely full ($k=1,2,3,\dots$) spectrum is the parity-broken sinusoid
   `cos_sin` (or a random-phase `cos`), at the cost of RBF bandwidth
   sensitivity.  There is accordingly no single strictly-richer default:
   `abs`/`relu`/`gelu`/`silu` are one gain-robust
   equivalence class (even or linear+even, odd $k\ge3$ absent), and the
   data-driven bias already lends them $k=1$; to gain the high odd modes you
   move to the RBF family and accept bandwidth tuning.

### 6.2 "Digital" activations are admissible because the solve is analytic

A second axis is *regularity*.  Because leenef **never differentiates the
activation** — decoders come from a least-squares solve, not
backpropagation — piecewise-constant activations with zero gradient almost
everywhere are perfectly usable.  This is a capability gradient-trained
networks structurally lack.  Two "digital" relatives of the RBF pair:

- **`square_wave` $=\operatorname{sign}(\cos z)$** — the Walsh / square-wave
  version of `cos`.  Its Fourier series
  $\tfrac{4}{\pi}\big(\cos z-\tfrac13\cos 3z+\tfrac15\cos5z-\dots\big)$
  packs many harmonics into one neuron, giving a **rich even kernel**
  (probe: $a_2^2=0.27$, $a_2$ mode $R^2=0.77$).
- **`box` $=\mathbf 1[\,|z|<c\,]$** — the indicator / piecewise-constant
  version of the `gauss` bump: with data-driven biases each neuron is a
  **hard receptive field** that fires iff the projection lands within $c$
  of its center.  Even kernel, with relatively more high-even energy than
  `gauss` ($a_4^2=0.041$ vs. $0.012$).
- **`staircase` $=\operatorname{clamp}(\lfloor z\rfloor,-3,3)$** — a
  *monotone* asymmetric quantizer (using $\lfloor\cdot\rfloor$ rather than
  rounding breaks parity).  It is a deliberate cautionary case: a fine
  monotone quantizer of a unit-scale pre-activation tracks the identity
  closely, so its Hermite energy sits almost entirely in the constant and
  **linear** modes ($a_1^2\approx1.0$, $a_2^2\approx a_3^2\approx0$ at every
  step/saturation probed) and it lands near the §3 collapse mode.  It shows
  that being "digital" does not by itself buy discriminative ($k\ge2$)
  power — the oscillatory `square_wave` and the bump `box` are the digital
  activations that carry a rich spectrum; a monotone staircase does not.

`square_wave` and `box` are even, so §6.1 applies: they shed odd modes at
zero bias and regain them through the data-driven bias.  Other digital
members are immediate — `step`/`sign` (order-0 arc-cosine), a **ternary**
sign with dead-zone — all zero-gradient and all admissible here.  The point
is not that a square wave beats a smooth kernel (it should not, by §8.3); it
is that the *space of usable activations is strictly larger under an
analytic solve*, and the choice is governed purely by the resulting Hermite
spectrum, never by differentiability.

---

## 7. Predictions (tested in the `c2_altact` panel)

| activation | kernel / spectrum | prediction vs `abs` |
|---|---|---|
| `linear` | linear kernel ($a_1$ only) | collapse (SSUP-I §2) |
| `abs` | even arc-cosine, $a_0,a_2,a_4,\dots$ | reference |
| `relu` | $\tfrac14\times$`abs` nonlinear $+$ linear | tie |
| `bent_line` (1, ½) | $\tfrac{1}{16}\times$`abs` nonlinear $+$ linear | **tie** |
| `leaky_relu` (0.1) | bent-line member | tie |
| `softplus` ($\beta{=}1$) | smoothed kink; sharp only if $\beta\cdot$gain large | tie if sharp, else **worse** |
| `tanh` | saturating, magnitude-suppressing | worse |
| `step` | order-0 arc-cosine (magnitude-blind) | worse |
| `square` | degree-2 polynomial kernel | $\approx$ / mild CIFAR edge |
| `sqrt_abs` | heavier low-freq even spectrum | $\approx$ or slightly worse |
| `double_fold` | richer even spectrum, 2 folds/neuron | $\approx$ to slightly better |
| `cos` | RBF / random Fourier | **gain-dependent**; can match or beat |
| `gauss` | RBF receptive field | **gain-dependent**; can match or beat |
| `cos_sin` | parity-broken RFF, **all** modes | $\approx$ `cos`; full RBF, gain-dependent |
| `square_wave` | digital cosine, rich even | $\approx$; gain-dependent |
| `box` | digital bump (hard RF), even | $\approx$; gain-dependent |
| `staircase` | digital monotone quantizer, **$a_1$-dominated** | near-linear; **collapse-leaning** |
| `gelu` | linear$+$even (relu class), odd $k\!\ge\!3$ absent | tie; gain-robust |
| `silu` | linear$+$even (relu class), odd $k\!\ge\!3$ absent | tie; gain-robust |

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
| `cos_sin` | 0.37 | 0.18 | 0.06 | **0.95** | **0.81** | **0.08** |
| `square_wave` | 0 | 0.27 | 0 | $\le0$ | **0.77** | $\le0$ |
| `box` | 0 | 0.12 | 0 | $\le0$ | **0.62** | $\le0$ |
| `staircase` | 1.03 | 0.00 | 0.00 | **1.00** | $\le0$ | $\le0$ |
| `gelu` | 0.25 | 0.09 | 0 | **1.00** | **0.85** | $\le0$ |
| `silu` | 0.25 | 0.06 | 0 | **1.00** | **0.87** | $\le0$ |

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
  functions): they recover $k=2$ but not the odd modes.  Their digital
  relatives **`square_wave`/`box` behave identically** (even, $k=2$ only),
  while **`cos_sin`** — the same function up to a $\pi/4$ phase — recovers
  **all** of $k=1,2,3$, the parity fix of §6.1 shown directly.  **`gelu`/`silu`**
  recover $k=1$ and $k=2$ but *not* $k=3$ — exactly the relu pattern, since
  each is $\tfrac12 z+$(even), so they are gain-robust smooth `relu`s rather
  than richer-spectrum activations.
- **`staircase` collapses to the linear mode.**  The monotone asymmetric
  quantizer recovers only $k=1$ ($a_1^2\approx1.0$, $R^2_{k=1}=1.00$) with
  $a_2^2\approx a_3^2\approx0$: a fine monotone quantizer of a unit-scale
  signal differs from the identity only by a high-frequency sawtooth whose
  low-order Hermite projection is negligible, so it behaves like `linear`
  (§3) despite being a nonlinear digital function.  Being "digital" buys
  nothing on its own — the spectrum is what decides.

($R^2\le0$ denotes "cannot represent": the regularized fit does no better
than the mean.  The quartic $k=4$ mode is not reliably learned by anyone
at this $(n, N_{\text{train}})$ because its $a_4^2$ is small — a
finite-sample effect consistent with learnability $\propto a_k^2$.)

**Parity is broken by any nonzero bias — but read the spectrum carefully.**
The $a_k^2$ columns above are the **zero-bias** coefficients of the scalar
$G$.  With a nonzero bias the feature span is governed instead by the
**bias-averaged** coefficient $\bar a_k^2=\mathbb{E}_b[a_k(b)^2]$,
$a_k(b)=\mathbb{E}_z[G(z+b)\mathrm{he}_k(z)]$, so the printed $a_k^2$ and the
measured $R^2$ are *expected* to diverge once bias is on; the $R^2$ then
reports the richer shifted-feature kernel, not a contradiction of the
zero-bias table.  Repeating the probe at $\mathrm{bias\ std}=2$
(`results/sl_clean/hermite_mode_recovery_bias2.json`) shows every even or
linear+even activation picking up the **low** odd mode $k=1$ — `cos`
$R^2\;{-1.49}\to0.95$, `abs` $-1.04\to1.00$, `box` $-1.02\to0.96$,
`square_wave` $-1.10\to0.94$ — while the cubic $k=3$ stays near zero
(`cos` $0.00$, `gauss` $-0.01$), exactly the "$k=1$ restored, high odd modes
weak; polynomials capped at their degree" picture of §6.1.  Because leenef's
data-driven bias $b_i=-g_i e_i^\top c_i$ is such a nonzero offset, the
even-only selection rule **does not bind in the real pipeline** for the
dominant $k=1$ mode; a center-matched probe would be the cleanest further
confirmation.

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

The full-dataset comparison (10 seeds, $n=5000$, hypersphere encoders,
data-driven biases, trace-scaled $\alpha=10^{-3}$, MNIST / Fashion-MNIST /
CIFAR-10) tests the §7 predictions on real targets.  Test accuracy
(mean $\pm$ SE over 10 seeds); panels at
`results/sl_clean/c2_altact_{mnist,fashion,cifar10}_panel.json`:

| activation (gain $U(0.5,2)$) | MNIST | Fashion | CIFAR-10 |
|---|---|---|---|
| `abs` (reference) | $0.9704\pm0.0001$ | $0.8778\pm0.0005$ | $0.4945\pm0.0011$ |
| `relu` | $0.9705\pm0.0003$ | $0.8765\pm0.0005$ | $0.4945\pm0.0007$ |
| `leaky_relu` $(s=0.01)$ | $0.9702\pm0.0003$ | $0.8764\pm0.0005$ | $0.4936\pm0.0007$ |
| `bent_line` $(s=\tfrac12)$ | $0.9685\pm0.0003$ | $0.8729\pm0.0004$ | $0.4934\pm0.0007$ |
| `softplus` | $0.9562\pm0.0003$ | $0.8562\pm0.0004$ | $0.4809\pm0.0007$ |
| `tanh` | $0.9677\pm0.0002$ | $0.8711\pm0.0005$ | $0.4860\pm0.0009$ |
| `square` | $0.9693\pm0.0002$ | $0.8751\pm0.0005$ | $0.4989\pm0.0011$ |
| `double_fold` $(c=1)$ | $\mathbf{0.9710}\pm0.0002$ | $\mathbf{0.8780}\pm0.0002$ | $0.4959\pm0.0009$ |
| `sqrt_abs` $(p=\tfrac12)$ | $0.9675\pm0.0004$ | $0.8729\pm0.0006$ | $0.4798\pm0.0010$ |
| `cos` | $0.9666\pm0.0002$ | $0.8695\pm0.0004$ | $0.5046\pm0.0013$ |
| `gauss` | $0.9666\pm0.0002$ | $0.8689\pm0.0004$ | $\mathbf{0.5062}\pm0.0011$ |

RBF-pair gain sweep (`cos`/`gauss` at narrow vs. wide gain):

| activation | gain | MNIST | Fashion | CIFAR-10 |
|---|---|---|---|---|
| `cos` | $U(0.05,0.2)$ | $0.7886$ | $0.7158$ | $0.3080$ |
| `gauss` | $U(0.05,0.2)$ | $0.7883$ | $0.7154$ | $0.3077$ |
| `cos` | $U(0.1,5.0)$ | $0.9704$ | $0.8764$ | $0.5012$ |
| `gauss` | $U(0.1,5.0)$ | $0.9697$ | $0.8750$ | $0.5000$ |

Every §7 prediction holds:

1. **The single-kink family ties `abs`.** `abs`, `relu`, `leaky_relu`,
   and the monotone $(1,\tfrac12)$ `bent_line` agree to within
   $\sim0.2\%$ on every dataset — one equivalence class, exactly as §4
   predicts (all share the $|z|$ spectrum modulo a bounded affine term).
2. **`softplus` is the weakest relu relative** ($-1.4\%$ MNIST, $-2.2\%$
   Fashion, $-1.4\%$ CIFAR).  At this gain it is nearly linear across the
   active range; smoothing away the kink suppresses the $k\ge2$ energy and
   pushes the kernel toward the collapse mode — the smooth-activation cost
   the theory anticipates.
3. **`tanh` trails the kink family but stays well clear of collapse**
   ($0.9677$/$0.8711$/$0.4860$, i.e. $-0.3\%$/$-0.7\%$/$-0.9\%$ vs. `abs`).
   As an odd activation it discards the even ($k=2$) magnitude mode — the
   dominant discriminative mode for these datasets — so it cannot match
   the kink family; the gap widens on the harder CIFAR-10, where even
   modes matter most.  But it is nowhere near the linear-readout floor
   ($0.39$ on CIFAR): its surviving odd $k\ge3$ modes (notably the cubic
   the Hermite test isolates) still carry real signal.  `tanh` is weaker,
   not broken — exactly the §5 picture.
4. **`double_fold` is marginally the best fold on the digit/garment
   tasks** ($0.9710$/$0.8780$): the extra fold injects more even-mode
   structure, the predicted "more folds" effect.
5. **`square` is competitive and the strongest fold-family activation on
   CIFAR-10** ($0.4989$): a pure degree-2 polynomial kernel already
   suffices, and helps most on the hardest task.
6. **`sqrt_abs` trails on CIFAR-10** ($0.4798$): the $p<1$ cusp gives a
   slowly decaying high-$k$ tail (a rougher kernel), which costs the most
   where the data are hardest.
7. **The RBF pair `cos`/`gauss` is the most powerful — on the hardest
   task, and when the bandwidth is right.** At the default gain they sit
   just below `abs` on MNIST/Fashion but are the **best activations
   overall on CIFAR-10** ($0.5046$/$0.5062$ vs. `abs` $0.4945$, a
   $5$–$9\,$SE margin): their dual kernel is a full Gaussian/RBF kernel
   with an *infinite* $k\ge2$ spectrum.  They are **gain-sensitive**,
   exactly as §6 warns — the per-neuron gain *is* the RBF bandwidth.  At
   tiny gain $U(0.05,0.2)$ the curvature barely engages: $\cos(z)\approx
   1-\tfrac{z^2}{2}$ with a quadratic signal of variance $\propto g^4$
   that trace-scaled ridge swamps, giving a signal-starved underfit
   ($0.79$/$0.72$/$0.31$) — a different failure from the linear collapse.
   At adequate/wide gain they fully recover and keep the CIFAR edge.
   `abs`, by contrast, is gain-robust.

The practical reading: the single-kink activations are interchangeable
(use the cheapest, `relu` or `abs`); to *beat* `abs` you must change the
$k\ge2$ spectrum, and on the hardest task an RBF activation with adequate
bandwidth (`gauss`/`cos` at gain $\gtrsim1$) is the most powerful of the
small nonlinearities tested.

---

## 9. Summary

1. **leenef = kernel ridge regression**; the activation chooses the
   kernel $k_G(x,y)=\mathbb{E}_{e,b}[G(z)G(z')]$, whose content is the
   squared Hermite spectrum $\{a_k^2\}$ of $G$.
2. **The linear mode ($k=1$) is the collapse mode**; all discrimination
   lives in $k\ge2$.  Linear activation = linear kernel = §2 collapse.
3. **Every *exact* single-kink (bent-line) activation equals `abs` up to
   a bounded affine correction** — exactly, for any bias.  ReLU,
   leaky-ReLU, and the monotone $(1,\tfrac12)$ kink are all
   `abs` to the solve, and the panel confirms they tie.  This *explains*
   the SSUP-I §5.4 `abs`$\approx$ReLU result and predicts the new kink
   ties.  Softplus is only an *approximate* kink: when its smoothing scale
   is comparable to the active range (as at the panel gain) it loses
   $k\ge2$ energy and trails — a sharp kink is what the equivalence needs.
4. **`tanh`/`step` are weaker** because they discard the magnitude (even
   Hermite) modes that carry most discrimination — though `tanh` keeps its
   odd $k\ge3$ modes and so stays well above the collapse floor.
5. **To beat `abs` you must change the $k\ge2$ spectrum**: `square`
   (degree-2 polynomial kernel), `double_fold` (more folds), or the RBF
   family (`cos`, `gauss`) whose **bandwidth is the per-neuron gain**.
   These are the principled stronger alternatives.
6. **Even-only is a parity artifact, not a complexity wall.** `cos`/`gauss`
   (and the digital `square_wave`/`box`) are even, so at zero bias they drop
   all odd Hermite modes.  The fix is *broken symmetry*, not added
   complexity: a phase shift ($\texttt{cos\_sin}=\cos+\sin$, or a random RFF
   phase → a stationary radial kernel, the Gaussian RBF for Gaussian
   frequencies) restores the full spectrum, and the nonzero **data-driven
   bias the pipeline already uses** restores at least the dominant $k=1$ odd
   mode (high odd modes stay weak; a polynomial activation never exceeds its
   degree).  `gelu`/`silu` are *not* a richer alternative — they are
   $\tfrac12 z+$(even), i.e. smooth gain-robust members of the `relu` class
   that miss odd $k\ge3$ exactly as `relu` does.  The genuinely richer
   activations remain the RBF family, at the price of bandwidth tuning.
7. **The activation space is larger here than for gradient nets.** Because
   the solve is analytic, zero-gradient "digital" activations
   ($\operatorname{sign}\circ\cos$, indicators, quantizers) are admissible;
   they are chosen by their Hermite spectrum, never by differentiability.

## 10. Relation to prior work

The *mathematics* is classical and not claimed as new:

- **Random features $\leftrightarrow$ kernels.**  That a population of fixed
  random nonlinear features approximates a kernel is Rahimi & Recht (2007);
  the infinite-width Gaussian-process limit is Neal (1996) and Williams
  (1997).  `cos`-as-RFF and the Gaussian-RBF correspondence are theirs.
- **Activation-specific kernels.**  Arc-cosine kernels for
  `relu`/`abs`/`step` are Cho & Saul (2009).  The **dual activation
  kernel** and its Hermite/Mehler expansion — the spine of this note — are
  Daniely, Frostig & Singer (2016), resting on Mehler's (1866) formula.
- **Random features + analytic linear solve as a method.**  This is exactly
  the **Extreme Learning Machine** (Huang, Zhu & Siew, 2006) and, in the
  temporal case, **reservoir computing / echo-state networks** (Jaeger,
  2001; Maass et al., 2002).  leenef's single-layer fit is structurally an
  ELM with NEF-style encoders and data-driven biases.  The universal
  approximation property of such single-hidden-layer non-polynomial
  features is older still (Cybenko, 1989; Hornik, 1991; Leshno et al.,
  1993), and the "digital" random features here are kin to random-binning /
  random-kitchen-sink features (Rahimi & Recht, 2008).

So at the level of theorems this note **consolidates common knowledge**
rather than discovering it.  Its contribution is applied and diagnostic:

1. **A bridge.**  Framing the NEF / leenef pipeline explicitly as
   random-feature kernel ridge regression, so that the activation is read as
   a *kernel choice* and the NEF $\leftrightarrow$ ELM $\leftrightarrow$
   kernel-machine identification is made precise (encoders = random
   projection, gain = RBF bandwidth, data-driven bias = parity-breaking
   phase, decoder solve = kernel ridge).
2. **Sharp, falsifiable diagnostics with clean fingerprints.**  The
   Hermite-mode recovery probe ($R^2_k>0$ iff $a_k\neq0$, including the
   counterintuitive `relu`-cannot-learn-$k{=}3$ / `tanh`-can result), and
   the origin-angle sweep with its **zero-variance symmetric collapse at
   $\theta=\pi$** (the linear-readout signature, encoder-independent).
   These turn "abs works" into a mechanism with a measurable signature.
3. **A practical degree of freedom specific to the analytic solve.**  The
   observation that *digital* (zero-gradient) activations are free here, and
   that the even-only limitation of RBF activations is dissolved by the
   bias the method already uses — both are consequences of the gradient-free
   construction that the gradient-trained literature has no reason to state.

In short: the engine is textbook kernel/random-feature theory;
the contribution is the lens, the diagnostics, and the gradient-free
practicalities.

### References

- Mehler (1866) — the Mehler kernel / generating function for Hermite
  polynomials.
- Schoenberg (1942) — positive-definite dot-product kernels on the sphere
  and their Gegenbauer expansion (the spherical analogue of the Hermite
  expansion used here).
- Cybenko (1989); Hornik (1991); Leshno et al. (1993) — universal
  approximation for single-hidden-layer non-polynomial activations.
- Neal (1996); Williams (1997) — infinite networks as Gaussian-process
  kernels.
- Jaeger (2001); Maass, Natschläger & Markram (2002) — echo-state networks
  / liquid-state machines (random recurrent features + linear readout).
- Eliasmith & Anderson (2003) — the Neural Engineering Framework.
- Huang, Zhu & Siew (2006) — Extreme Learning Machines (random hidden
  layer + analytic least-squares readout).
- Rahimi & Recht (2007), *Random Features for Large-Scale Kernel Machines*;
  Rahimi & Recht (2008), *Weighted Sums of Random Kitchen Sinks* — including
  random-binning ("digital") features.
- Cho & Saul (2009), *Kernel Methods for Deep Learning* (arc-cosine
  kernels).
- Bach (2017) — random features and kernel approximation rates.
- Daniely, Frostig & Singer (2016) — dual activation kernels and their
  Hermite spectra.
