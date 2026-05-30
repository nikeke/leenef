# Abs-activation preimages and class prototypes

Exploratory study of what the absolute-value activation does to the
input → activity → class map in a single default-config `leenef`
`NEFLayer`, and what that implies for generating *class prototypes* and
*preimages*.

Branch base commit: `ba09221459d53751c1823b7e243b01511af05987`.
All numbers below are CPU, MNIST, fixed seed 0, 10000-sample train
subset, unless noted.

---

## 0. The original question

> Because of our abs activation we might get $2^{n}$ "perfect"
> prototypes by taking one or the other sign?

The short answer: **yes, but only when `n_neurons` $\le d_{\mathrm{in}}$.**
In the realistic expansion regime (`n_neurons` $> d_{\mathrm{in}}$) the
sign freedom is mostly *unrealizable*, yet a different mechanism — the
decoder null space — still yields a high-dimensional continuum of exact
preimages. The rest of this note makes both statements precise and
visualizes them.

---

## 1. Math background: a NEF layer is affine-then-abs

For input $x \in \mathbb{R}^{d_{\mathrm{in}}}$ the default layer computes

$$
z = W x + b, \qquad a = |z|, \qquad \hat y = D^\top a,
$$

where

- $W = \operatorname{diag}(g)\,E \in \mathbb{R}^{n \times d_{\mathrm{in}}}$
  stacks the per-neuron gain $g_i$ times the fixed encoder direction
  $e_i$ (rows of $E$);
- $b \in \mathbb{R}^{n}$ is the (data-driven) bias,
  $b_i = -g_i\,(d_i \cdot e_i)$;
- $a = |z|$ is the **abs** activation, applied coordinate-wise;
- $D \in \mathbb{R}^{n \times n_{\mathrm{classes}}}$ are the analytically
  solved decoders.

The only nonlinearity is the elementwise $|\cdot|$. Therefore the layer
is **continuous piecewise-linear**: the input space is partitioned into
*sign cells*

$$
\mathcal{C}_s = \{\, x : \operatorname{sign}(Wx+b) = s \,\},
\qquad s \in \{\pm 1\}^{n},
$$

and inside one cell abs acts as the fixed diagonal sign $\operatorname{diag}(s)$,
so

$$
a(x) = \operatorname{diag}(s)\,(Wx+b),
\qquad
\hat y(x) = \underbrace{D^\top \operatorname{diag}(s) W}_{=: \,M_s}\;x
            + D^\top \operatorname{diag}(s)\,b .
$$

Both the encode $x \mapsto a$ and the full decode $x \mapsto \hat y$ are
**affine within each cell** with a cell-dependent matrix.

### 1.1 Why abs is different from ReLU / a generic net

- **Sign symmetry.** $|z_i| = |-z_i|$. The activation is *even*, so each
  neuron throws away exactly one bit — the sign of its pre-activation.
  ReLU instead *clamps* negatives to $0$ (it is not invertible on the
  negative half-line), and a generic smooth net mixes coordinates
  nonlinearly. Abs uniquely preserves magnitude while discarding sign,
  which is what makes the "flip a sign, get another preimage" idea even
  conceivable.
- **Exact, enumerable preimages.** Because the map is piecewise-*linear*
  (not just piecewise-smooth), preimages of a target activity or a target
  logit are solutions of *linear* systems — recoverable in closed form
  per cell, with no iterative inversion. A generic network has no such
  closed form.
- **Combinatorial preimage structure.** A target activity $a^\* \ge 0$
  has up to $2^{n}$ algebraic sign-preimages $z = s \odot a^\*$. With
  ReLU there is no analogous sign multiplicity (a zero activation hides
  *all* negative pre-activations in one ray, a positive activation pins
  the value exactly). Abs is the activation that maximizes this
  preimage multiplicity.

---

## 2. Sign-preimages of a target activity (`sign_freedom_experiment.py`)

Pick a target activity $a^\* \ge 0$. Each sign vector $s$ gives a
candidate pre-activation $z = s \odot a^\*$ and hence a candidate input
by least-squares solving $W x = s \odot a^\* - b$. The recovered $x$
*re-encodes to exactly* $a^\*$ iff $s \odot a^\* - b \in \operatorname{col}(W)$, a
$d_{\mathrm{in}}$-dimensional subspace of $\mathbb{R}^{n}$.

| regime | natural-sign recon err (real sample) | random-flip recon err | random flips decoding to class |
|---|---|---|---|
| `n=400`  $(\le 784)$ | $1.2\times10^{-6}$ | $\sim 2\times10^{-6}$ | **200 / 200** |
| `n=2000` $(> 784)$  | $1.7\times10^{-6}$ | $\sim 0.61$ | 35 / 200 |

- **Underdetermined `n` $\le d_{\mathrm{in}}$:** $W$ has full row rank, so
  $Wx = t$ is solvable for *every* $t$. All $2^{n}$ sign flips are
  exactly realizable; every one re-encodes to $a^\*$ to machine precision
  and decodes identically. The original guess holds literally — there are
  $2^{n}$ **perfect twins**.
- **Overdetermined `n` $> d_{\mathrm{in}}$ (realistic):** activities live
  on a thin $d_{\mathrm{in}}$-dimensional folded manifold inside
  $\mathbb{R}^{n}$. A *real* sample's natural signs are exactly realizable
  (err $\sim10^{-6}$), but flipping signs pushes the target off the
  manifold (relative reconstruction error $\sim 0.6$), and only a minority
  of flips still happen to decode to the right class.

### Figure: `figures/perfect_twins.png`
Underdetermined `n=400`: a real digit plus five sign-flipped
reconstructions. The five "twins" are visually unrelated noise images,
yet their activity vectors match the real digit's to a max deviation of
$\sim 4\times10^{-6}$ and they all produce the **identical** decode. This
is a clean illustration of how much information abs discards: one sign
bit per neuron, here $400$ bits, all invisible to the layer.

---

## 3. Going back from the one-hot result (`visualize_preimages.py`)

The user's sharper question: in a realistic overdetermined network, if we
invert the *one-hot class target* back to input space, do we get exactly
one realizable preimage, or none?

The full inverse stacks two maps: $y^\* \to a \to x$.

1. **The naive activity inverse is not realizable.** The min-norm
   activity $a_0 = (D^\top)^{+} y^\*$ has on average **~29–31 % negative
   entries**, so no $x$ can produce it ($a = |z| \ge 0$). You cannot
   simply invert the decoder and then the encoder.

2. **The piecewise-linear cell gives a continuum, not a point.** Fix a
   sign cell $s$. Inside it the decode is $M_s x + D^\top(s\odot b) = y^\*$,
   i.e.

   $$
   \underbrace{D^\top \operatorname{diag}(s) W}_{M_s \in
   \mathbb{R}^{n_{\mathrm{classes}} \times d_{\mathrm{in}}}}\; x
   = y^\* - D^\top(s \odot b).
   $$

   That is $n_{\mathrm{classes}} = 10$ equations in $d_{\mathrm{in}} = 784$
   unknowns — an **affine family of preimages of dimension
   $d_{\mathrm{in}} - n_{\mathrm{classes}} = 774$ per cell.**

So the answer to "just one, or none?" is **neither: generically a
774-dimensional infinity of realizable preimages**, supplied by the
decoder's null space.

### 3.1 Per-class preimages — `figures/overdet_preimages.png`

Anchoring the smallest in-cell correction to each class mean,
$x = \bar x_c + M_s^{+}(y^\* - M_s \bar x_c)$, recovers a recognizable
digit-like prototype that decodes to **exactly** its one-hot target
(softmax $p = e/(e+9) \approx 0.23$, the value of a perfect one-hot logit
under 10-way softmax). Note the caveat: the *min-norm-from-origin*
solution instead escapes its assumed sign cell and decodes to garbage —
the cell linearization is only valid locally, so the inversion must be
anchored near real data.

### 3.2 The null-space continuum — `figures/nullspace_continuum.png`

Walking the $774$-dimensional null space of $M_s$ from an anchor produces
visibly different "3" images that **all** keep class 3 with one-hot
decode deviation $\le 0.09$. The realizable preimage set of a one-hot is
a high-dimensional affine family, not a point — a direct visualization of
the decoder null space restricted to the input image.

---

## 4. Canonical prototypes and the margin↔realism tradeoff (`canonical_prototypes.py`)

The exact-one-hot preimage only asks $D^\top a = e_c$ and so decodes at
the unremarkable confidence $p \approx 0.23$. A **canonical prototype**
instead maximizes the decision margin

$$
m_c(x) = \text{logit}_c(x) - \log\!\sum_{j \ne c} e^{\text{logit}_j(x)},
$$

over the pixel box $x \in [0,1]^{d_{\mathrm{in}}}$ (sigmoid
reparametrization), optionally with a manifold anchor
$\lambda\,\lVert x - \bar x_c\rVert^2$. Because the forward map is the
closed-form $\,|Wx+b|\,D$, this is the NEF analogue of
activation-maximization / feature visualization.

| anchor $\lambda$ | mean margin | mean $\lVert x-\bar x_c\rVert$ | argmax = class |
|---|---|---|---|
| (class mean) | $-1.43$ | $0.00$ | 100 % |
| $0.0$  | $+6.02$ | $14.49$ | 100 % |
| $0.02$ | $+4.68$ | $9.83$  | 100 % |
| $0.05$ | $+2.35$ | $5.41$  | 100 % |
| $0.10$ | $+0.93$ | $3.16$  | 100 % |
| $0.20$ | $-0.06$ | $1.74$  | 100 % |

**Key finding — margin and realism are in direct conflict.**
Unconstrained margin maximization ($\lambda = 0$) yields high-margin
($+6$) *adversarial noise* sitting $\sim 14$ away from the class mean;
strengthening the anchor morphs the prototype back into a recognizable
digit but drives the margin to $\approx 0$. Note that the class mean
itself has a *negative* margin ($-1.43$) despite being classified
correctly — a competitor class's log-sum-exp slightly exceeds the true
logit. The confident regions of the decoder live **off** the data
manifold: with random encoders the layer behaves like a classic
adversarial-prone linear model in activity space.

### Figures
- `figures/canonical_prototypes.png` — a grid morphing from class means
  (top) through adversarial noise ($\lambda=0$) to digit-like templates
  ($\lambda=0.2$), with per-image margins annotated.
- `figures/margin_realism_tradeoff.png` — the monotone Pareto frontier of
  mean margin vs distance-from-mean.

---

## 5. How far does the continuum extend? (`nullspace_radius_sweep.py`)

Starting from an exact one-hot preimage $x_{\mathrm{part}}$ of class 3, we
step a radius $r$ along (a) random **null-space** directions of $M_s$ —
the realizable continuum — and (b) random **ambient** directions, and
average over 40 random unit directions per radius.

`figures/nullspace_radius_sweep.png` shows three panels:

1. **decode deviation** grows more slowly for null-space moves than
   ambient moves;
2. **decision preserved**: null-space moves keep class 3 for the entire
   tested range (class-flip radius $r_{0.5} = \infty$ within $r \le 12$),
   while ambient moves flip the decision at $r_{0.5} \approx 10$;
3. **abs folds crossed**: null-space and ambient moves cross *the same
   fraction* of sign boundaries.

The third panel is the subtle one: null-space steps trip just as many
sign flips as ambient steps, but those flips land on **decode-irrelevant
neurons** (near their zero-crossing, where $a_i \approx 0$, so $\pm a_i$
barely changes the logit). The decision therefore survives far longer
even though the layer is technically leaving its original linear cell at
the same rate. In other words, the realizable continuum is *robust*, not
merely *exact-in-cell*.

---

## 6. Reproducibility

```bash
source venv/bin/activate
cd explorations/abs_preimages
python sign_freedom_experiment.py     # console table (section 2)
python visualize_preimages.py         # perfect_twins, overdet_preimages, nullspace_continuum
python canonical_prototypes.py        # canonical_prototypes, margin_realism_tradeoff
python nullspace_radius_sweep.py      # nullspace_radius_sweep
```

- All randomness uses explicit `torch.Generator` seeds; encoders/gains/
  biases are functions of the seed only (default-config layer built by
  `common.build_layer`).
- `common.py` adds `benchmarks/sl_clean` and `src` to `sys.path`, loads
  MNIST via the harness loader (raw-byte hashed), and exposes the affine
  pieces $W, b$ directly.
- These are exploratory single-seed scripts, not part of a Paper-I/II
  multi-seed sweep; treat the numbers as illustrative of the *mechanism*,
  not as benchmarked claims.

---

## 7. Ideas to keep exploring

**Geometry / theory**

1. *Sign-cell census.* For the overdetermined layer, count how many of
   the $2^{n}$ sign cells are actually occupied by the data manifold, and
   measure typical cell diameters around real samples. The "twins"
   combinatorics is bounded by occupied cells, not $2^{n}$.
2. *Exact realizable-flip set.* Characterize which single-bit sign flips
   keep $s \odot a^\* - b \in \operatorname{col}(W)$. These are the flips on neurons
   whose $e_i$ is (near) orthogonal to the complement of $\operatorname{col}(W)$;
   enumerate them to get the *true* local preimage multiplicity in the
   expansion regime.
3. *Half-quadratic / square activation contrast.* Repeat the whole study
   with `square` (also even) and `relu` (clamping) activations to isolate
   which phenomena are specific to abs vs to evenness vs to
   piecewise-linearity.

**Prototypes / generation**

4. *Manifold-faithful prototypes.* Replace the L2-to-mean anchor with a
   stronger data prior (PCA-whitened penalty, a learned density, or a
   small denoiser) to get prototypes that are simultaneously high-margin
   *and* realistic — testing whether the conflict in §4 is intrinsic or
   just a weak-prior artifact.
5. *Per-cell margin LP.* Within a fixed sign cell the margin is linear in
   $x$, so maximizing the *worst-case* (nearest-competitor) margin over
   the box is a max–min linear program with a closed structure. Solve it
   exactly and compare to the gradient prototypes.
6. *Ensemble / multi-layer prototypes.* Extend to `NEFEnsemble` and
   `NEFNetwork` (encode-only hidden layers): does averaging over random
   encoders wash out the adversarial noise and make canonical prototypes
   recognizable without a prior?

**Robustness / security**

7. *Twins as adversarial pairs.* In the underdetermined regime the $2^{n}$
   twins are *exact* adversarial collisions (identical activity, arbitrary
   input). Quantify how `n_neurons` relative to $d_{\mathrm{in}}$ trades
   off accuracy against collision volume — a direct knob on the
   "information bottleneck" of the abs fold.
8. *Decoder-null-space attacks.* The 774-dim null-space continuum is a
   ready-made family of confident-but-arbitrary inputs; measure how small
   a null-space perturbation must be to remain visually on-manifold while
   flipping a *downstream* model's prediction.

**Applications**

9. *Preimage-based data augmentation.* Sample small in-cell null-space
   perturbations of real digits (which provably keep the exact label) as
   a label-preserving augmentation, and test whether retraining decoders
   on them improves generalization or calibration.
10. *Class-prototype initialization.* Use the anchored one-hot preimages
    (§3.1) as data-driven `centers=` for a second NEF layer, closing the
    loop between "prototype generation" and "encoder placement".
