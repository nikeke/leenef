# Paper I (Supervised Learning) — accumulated insights

This file collects interpretive observations, theoretical framings, and
"things to mention when we write" that won't fit cleanly into the plan or
the result JSONs.  Update this as we discover or sharpen things.  When it
is time to write, this file is the first thing to re-read.

## 0pre. The six principles of leenef (the trilogy's shared opener)

Distilled from the program's working code (`streaming.py`,
`layers.py`'s `partial_fit` / `solve_accumulated` / `continuous_fit`,
`recurrent.py`, `rl.py`) and from the sparse-reward-c plan.  These are
the operational claims; everything else in Paper I is one of them
applied somewhere new.  Papers II and III repeat exactly this
distillation as their opener, so readers picking up any volume meet
the same six lines first.

1. **Encoders are fixed** (random or data-adapted).  Decoders are the
   only thing we ever fit.
2. **Decoders fall out of one analytic step**: `D = (AᵀA + αI)⁻¹ AᵀY`.
   No learning rate, no schedule, no gradient flow between layers.
3. **`AᵀA` and `AᵀY` are the universe.**  Continual learning,
   streaming over time, ensembles, online updates, per-action FQI —
   all of them are different ways of *additively combining*
   sufficient statistics.
4. **Targets `Y` are the only place where task structure enters.**
   The encoder doesn't care what task it is; the decoder column does.
5. **Time is collapsed by pooling, not by recurrence.**  The streaming
   classifier mean-pools delay-line activities into a fixed-size
   representation; no BPTT, no recurrent solve.
6. **Per-class additivity is what makes CL work.**  Each class owns
   its own decoder column, so accumulating new task statistics never
   interferes with old columns. (Forward-look — paper II.)

Whenever a leenef method has felt natural rather than bolted-on, it
has done so by re-using one of these primitives in a new place.
Whenever something has fought back (recurrent gradient-free, deep
gradient-free hierarchies, sparse-reward RL), it has done so because
one of these principles broke.

For Paper I specifically: §3 (method) is principles 1–4, §5.7
(streaming classifier) is principle 5, and §5.5 (multi-layer) is the
load-bearing negative result.  C4 (`results/sl_clean/c4_multilayer_mnist_panel.json`,
5 seeds, MNIST, hidden=2000, output=2000, abs, hypersphere,
data-driven biases, α=10⁻³) refines the negative: greedy 95.37 ± 0.09 %,
hybrid 94.25 ± 0.04 %, target_prop 98.54 ± 0.02 %,
end_to_end 98.60 ± 0.02 %, hybrid_e2e 98.69 ± 0.01 %.  Plain
analytic stacking (greedy/hybrid) loses 3–4 pp; layer-local
target propagation recovers single-layer accuracy without
inter-layer gradient flow; full E2E nudges another ~0.1 pp.
The trilogy still stays single-layer because the depth gain is
small, but the headline reads "purely analytic depth does not
help" rather than "depth never helps".

## 0. Why over-representation needs nonlinearity (CORE — open the paper with this)

A linear activation collapses the entire scheme.  If `a_i = α_i (e_i · x)`,
then for any decoder `D ∈ ℝ^{n × d_out}`,

    y = D^T a = D^T diag(α) E x = (D^T diag(α) E) x = W x

is a single linear map, regardless of `n`.  Over-completeness buys
**nothing**: a 100000-neuron linear network is exactly as expressive as a
single linear projection of input onto output.  The frame `{α_i e_i}` is
just an absurdly redundant parametrisation of a `d_in × d_out` matrix.

The *minimal* repair is a pointwise nonlinearity that is not
representable as a linear scaling — and `abs` is the smallest such repair
we can imagine:

    a_i = |α_i · e_i · x|

Geometrically, `|e_i · x|` is the **unsigned distance of x from the
hyperplane `{e_i · z = 0}`**.  Each neuron measures one such unsigned
distance.  A linear readout over n unsigned-distance measurements is a
sum of folded half-planes, i.e. a piecewise-linear function with up to
`n` fold lines.  With n large and `e_i` covering directions densely
enough, this function class is universal on compact subsets of ℝ^d.

ReLU is the same idea after a half-fold (drop the sign instead of fold
about the origin).  Same expressivity story, slightly different
geometry.  Other smooth nonlinearities (tanh, sigmoid, GELU) trade the
sharp fold for a soft one — same theoretical universality, sometimes
better gradient behavior for end-to-end training, but no advantage when
training is analytic and the linear-collapse argument is what we're
working against.

**Why this is the right thing to open the paper with:**
- It explains the existence question.  Why does `NEFLayer.fit` work *at
  all* without learned features?  Because abs already gives you piecewise
  linearity; the only thing left to learn is which piecewise-linear
  function to assemble.
- It motivates random encoders.  Once you accept that the frame is a
  set of fold directions, the precise placement of folds matters less
  than their coverage.  Random uniform directions (`hypersphere`) are
  near-optimal coverage; data-adapted directions (`whitened`,
  `local_pca`) are coverage tilted toward the data manifold.
- **It explains where bias placement matters (§1.2).**  Linear
  activations would *absorb* any constant offset into the decoder and
  forget.  abs doesn't, because the fold point is where
  `α_i (e_i · x) + b_i` crosses zero — and that fold sits in input
  space exactly where the bias places it.  When the data centroid
  sits far from that hinge, the data rarely crosses zero and abs is
  effectively linear; when the data sits *on* the hinge, half the
  mass falls on each side and abs collapses sign symmetrically.  So
  §1.2's "where you put the hinge matters" and §0's "abs is the
  minimal nonlinearity needed for over-representation to do work"
  are the same fact viewed from two angles.
- It is the cleanest answer to the obvious-skeptic question: "your
  decoder is linear; what makes this not just feature engineering plus
  linear regression?"  The honest answer is that yes, it *is* feature
  engineering plus linear regression — and the feature engineering is a
  fixed, mathematically minimal nonlinearity that already makes the
  function class universal.  Everything else in the paper is asking how
  far that simple combination can be pushed.

The whole NEF programme — random/data-adapted encoders, analytic
solves, no gradient training — is buying you the simplest computational
artefact that escapes the linear collapse.  Once you have the fold,
linearity downstream is *enough*.

## 1. The frame-coefficient interpretation (CORE)

For our default `abs` activation and zero (or near-zero) bias, the per-
neuron activation reduces to

    a_i = |α_i · e_i · x|

which is the **absolute value of the frame coefficient of x in the
overcomplete frame {α_i · e_i}_{i=1..n}**.  Mean-pooling these absolute
frame coefficients over a delay-line window gives a fixed nonlinear
embedding; the linear decoder then learns a task-specific readout from
that frame.

Implications to develop in the paper:

- Decoupling frame design from learning.  Encoder strategies
  (`hypersphere`, `gaussian`, `sparse`, `receptive_field`, `whitened`,
  `class_contrast`, `local_pca`) are all ways of choosing the frame.
  Random-direction frames are universal but data-blind; data-adapted
  frames trade universality for efficiency.
- The role of `gain`.  A per-neuron `α_i` rescales each frame element;
  this is equivalent to non-uniform measurement of x.  The default
  U(0.5, 2.0) range gives a mild adaptive scaling that empirically beats
  α≡1.
- The role of `bias`.  A non-zero `b_i` translates the input before the
  absolute value: `a_i = |α_i · e_i · (x − d_i)|` where `d_i = b_i · e_i`.
  In the data-driven-bias setting, `d_i` is a training sample, so each
  neuron measures the magnitude of x's projection onto e_i *as seen from
  d_i* — a local frame coefficient.  When `b_i = 0`, all neurons share
  the origin; the frame becomes globally-centered.  B1's biases ablation
  is a direct test of whether local centering buys anything over the
  globally-centered frame.

### Manifolds and dataset centeredness

Real data lives on (or near) a low-dimensional manifold inside the
ambient input space.  Two complementary manifold-flavoured roles for
the ingredients above:

- **Random encoders** approximate a near-isometric measurement of the
  manifold (Johnson-Lindenstrauss / restricted isometry style):
  enough random directions in high dimension preserve pairwise
  distances and angles, so the data manifold's geometry is carried
  faithfully into the n-neuron frame coefficients.  The decoder then
  reads out a task variable from that geometry.
- **Data-driven biases** place each neuron's measurement origin on the
  data manifold itself.  Each neuron answers "how far is x from this
  particular manifold sample, *along this direction* e_i?".  This turns
  the global frame into a manifold-aware *local* frame: tangent-style
  coordinates from many neighborhoods at once.

Whether *local* (data-driven) framing actually helps over *global*
(zero-bias) framing depends on the dataset's relationship to the
fold point of `abs`:

- The activation is `a_i = |α_i · e_i · (x − d_i)|`.  The fold point
  is the hyperplane `e_i · x = b_i / α_i` (or just `e_i · x = 0` for
  zero biases).  Inputs whose projection onto e_i straddles this fold
  point lose sign information after the absolute value; inputs whose
  projection sits firmly on one side of the fold preserve their
  geometry up to a scalar.

#### Where the abs hinge sits relative to data (POST-B7 REFRAMING)

The headline of B7 is geometric, not spectral.  We initially read the
B1 split (MNIST/Fashion: data-driven bias buys nothing; CIFAR-10:
buys ~1.4 pp) as a DC-component story — DC dominates → abs effectively
linear → zero bias is fine; small DC → abs folds → data-driven bias
helps.  After running the centered-CIFAR-10 and decentered-MNIST
follow-ups (10 seeds × 9 cells each), the spectral framing is
falsified.  The mechanism is **where the abs hinge is placed relative
to the data centroid**, and DC is one specific instance of it.

|       scenario       |  DD%   | zero%  |  gap    | ‖data centroid‖ |
|----------------------|--------|--------|---------|-----------------|
| MNIST raw            | 97.08  | ~97.0  |  0.0    | far from 0      |
| MNIST decentered+0.5 | 97.12  | 95.78  | **+1.3**| farther from 0  |
| CIFAR-10 raw         | 57.79  | 56.40  |  +1.4   | moderate        |
| CIFAR-10 centered    | 49.72  | 43.41  | **+6.3**| AT 0            |

The clean reading: with abs activation `a_i = |α_i (e_i · x) + b_i|`,
the bias `b_i` selects where the hinge sits in input space.  Zero
bias places every hinge through the origin.  Data-driven bias
`b_i = -α_i (e_i · d_i)` places the i-th hinge at training sample
`d_i`, anchored on the data manifold.

- If the data centroid is *at* the origin, zero bias is a worst case:
  half the data mass falls on each side of every hinge and abs
  collapses sign symmetrically.  CIFAR-10 centered is exactly this —
  the +6.3 pp gap is the largest we've seen.
- If the data centroid is *far from* the origin, zero bias is
  accidentally sane (almost no sign flips) and the abs is effectively
  linear in the data fluctuations.  MNIST raw is exactly this — gap
  ≈ 0.
- In between (CIFAR-10 raw, MNIST decentered), the gap scales
  roughly with how often `e_i · x` actually crosses zero relative to
  its typical magnitude.

So **there is one axis, not three regimes.**  The original DC reading
got the prediction direction wrong on CIFAR-10 (we expected centering
to *close* the gap; it *opens* it the most).  The geometric reading
predicts both halves of the table correctly: gap grows with the
distance between zero and data mass.

We retain the DC vocabulary in the paper as a useful introduction
("zero-bias = hinge through origin; if the data has a non-zero mean,
that's a constant offset that abs would fold around") but the headline
mechanism is "the abs hinge is a hyperplane; where you put it
matters; placing it on the data manifold is the natural choice for
overcomplete decoding."  See `results/sl_clean/biases_*_panel.json`
for the full B7 data and §7/H7 below for the hypothesis ledger.

Connection to compressive sensing / sparse coding.  Random frames with
enough redundancy approximately preserve geometry (Johnson-
Lindenstrauss / restricted isometry); abs-rectified frame coefficients
are a practical proxy for measurement magnitude.  Data-driven biases
introduce a manifold-local refinement on top of this.  Both threads
converge on the same picture: **the encoder layer is a fixed manifold-
aware frame; only the readout is task-specific**.

## 2. Regularization must scale with n_neurons (CORE)

The Tikhonov solve `(AᵀA + reg I) D = AᵀY` becomes meaningless when `reg`
is set without reference to the trace of `AᵀA`.  As n_neurons grows, the
average diagonal mass per neuron stays roughly constant (for our
normalized inputs and abs activations), so the *total* trace grows
~linearly in n.  A fixed `α=1e-2` therefore becomes negligible at large
n: the solve is effectively unregularized.

What we observed (B2 α calibration, fixed α=1e-2):
- n=2000: 97.30 ± 0.04
- n=4000: 98.07 ± 0.03
- n=8000: 87.27 ± 6.77, with one seed below 80% and another at 85%

What every other solver in our codebase already does (`solvers.tikhonov`,
`solve_from_normal_equations`, `normal_equations`):

    reg = max(α · trace(AᵀA) / n_neurons,  α)

The streaming module was the inconsistent outlier.  Once we patch the
experiment runner to use the same trace-scaled reg, n=8000 becomes well-
behaved.

This is a paper-worthy lesson:
- **Pedagogical.**  Beginners tune α as a "magic constant" and are then
  bitten by silent under-regularization at scale.  Trace-scaling is the
  principled default.
- **Negative-result evidence.**  The fixed-α baseline (preserved under
  `results/sl_clean/fixed_alpha_baseline/`) is the cautionary tale.  Our
  trace-scaled re-run is the resolution.  The contrast is the figure.
- **Connection to the frame story.**  The per-neuron diagonal of `AᵀA` is
  the squared norm of that neuron's column of activities, i.e. the
  *energy* deposited in that frame element across the dataset.  Trace-
scaling normalizes α by the average frame-element energy, which is the
  geometrically natural choice.

### 2b. The α multiplier itself depends on the architecture (CORE)

A second over-correction lesson.  Once we trace-scaled with the
"standard" α=1e-2 (the same number `solvers.tikhonov` uses for plain
`NEFLayer.fit`), streaming *under-performed by ~2 pp at n=4k*.

What we observed (B2, trace-scaled α=1e-2, alpha_eff ≈ 115):
- n=4000 solve_modes panel: 0.9588 ± 0.0006 (vs old fixed-α 0.9806)

Why streaming differs from `NEFLayer.fit` even though both use the same
trace formula: in plain `NEFLayer.fit` the per-sample diagonal scale is
small (typical |projection|² ~ O(0.1)) and reg = α · 60000 · 0.1 ≈ 60
sits below the data spectrum.  In streaming, mean-pooling preserves
abs-projection magnitudes, so trace/n ≈ 11000 and reg = α · 11000 ≈
110 — large enough to bite the leading singular components.

The right answer is **a small α multiplier** with trace-scaling (or
equivalently a smartly chosen fixed α at this n).  We measure it
directly with `streaming_alpha_sweep`:
- 5 seeds × 5 alphas {1e-6,1e-5,1e-4,1e-3,1e-2}
  × 2 alpha_scales {fixed, trace}
  × 2 n_neurons {4000, 8000}
- The intersection of the two scaling laws gives us:
  - the fixed-α minimum at each n (and how it shifts with n)
  - the trace-α minimum at each n (and how flat it is across n)
  - the optimal alpha_eff and how it scales with n

This is itself a paper sub-result.  The headline figure is a 2×2 grid
of accuracy-vs-α curves at (fixed/trace) × (n=4k/n=8k) showing both
failure modes and identifying the geometric sweet spot.

The streaming default we settle on becomes a documented choice in the
paper, with this sweep as evidence.  No magic numbers.

## 3. H3: solver-numerics equivalence

Three solve modes give identical accuracies (within 0.01 pp) when given
the same accumulated AᵀA / AᵀY and the same alpha_eff:

- `batch_acc`:  one-shot full-data accumulate + Cholesky solve.
- `accumulate`: chunked accumulate (multi-batch normal equations) + solve.
- `woodbury`:   `continuous_fit` rank-k updates of M⁻¹.

Verdict: **equivalence supported**.  The choice of solve strategy is a
deployment concern (memory / online vs batch), not a statistical one.

For online/continuous learning the Woodbury path is the only one that
delivers up-to-date decoders after each batch without re-solving.  For
batch settings, `accumulate + solve` is the simplest and fastest.  This
is the "infrastructure" answer the paper needs to give *before* it
starts comparing to gradient-based baselines.

### Numerical-stability and GPU implications

`accumulate + solve` and `continuous_fit` (Woodbury) sit at different
points on the precision/throughput spectrum, and the difference matters
once we move to GPU:

- `accumulate + solve`: GEMMs (`AᵀA`, `AᵀY`) run in **float32** for
  speed; the running stats are accumulated in **float64** to absorb
  catastrophic cancellation across many batches.  The final solve runs
  in float64.  Memory cost is the n×n / n×d_out stats only.
- `continuous_fit` (Woodbury): the n×n inverse `M⁻¹` must be maintained
  end-to-end in **float64** to keep rank-k updates from drifting.
  Memory cost includes the explicit n×n inverse.  On GPU at large n
  this is the limiting factor; refresh-by-exact-inverse periodically
  helps but defeats some of the Woodbury speed advantage.
- Practical guidance: on CPU at moderate n, all three are fine.  On GPU
  with large n, prefer `accumulate + solve` unless you genuinely need
  per-batch up-to-date decoders.  This is a real constraint and we
  should state it explicitly in the paper.

## 4. Limits of the delay-line + abs + mean-pool architecture

sMNIST-row works because spatial locality (28-pixel rows are coherent
short-term temporal patterns).  sMNIST-permuted destroys that locality
and the same architecture collapses to 22-25%, regardless of n_neurons in
{2k, 4k}.  This is an honest published limit — *not* a failure to be
hidden.  The architecture is a measurement device for short-time-locally-
coherent input streams; permuted pixels are not such a stream.

### Mitigations to attempt (and probably watch fail)

We won't just declare the limit and walk away.  A few cheap mitigations
are worth running so the paper documents *what the architecture cannot
absorb*, not just *what we didn't try*:

- **Larger windows.**  K ∈ {10, 20, 50, 100, 784}.  As K → 784 the
  mean-pool sees the full sequence; if the issue is myopic locality,
  bigger K should help.  Likely fail: mean-pooling absolute frame
  coefficients destroys order, so even K=784 cannot recover position.
- **Position encoding.**  Concatenate a per-timestep positional
  feature (or a small learned positional embedding) to the input
  before the delay line.  This restores order information and turns
  the delay-line into something closer to an unordered-bag-of-pixels
  classifier with positional tags.  Likely the strongest mitigation,
  though it stops being "honest delay-line streaming".
- **Replacing mean-pool with last-state pool / max-pool.**  Mean-pool
  is what kills order; max-pool over the K-window keeps only the
  largest excursion and may behave more like a salience filter.
  Probably small help on permuted, but worth a quick check.
- **Data-adapted encoders (e.g. `local_pca` on permuted patches).**
  Lets the frame adapt to the broken-up data distribution.  May help
  if the issue is encoder mismatch rather than architecture mismatch.
  Likely fail: the bottleneck is "no spatial locality at all".

We expect all of these to either fail outright or to mostly recover
the unordered-bag-of-pixels MLP baseline (~96% on sMNIST-permuted with
~4k features).  Either way, the *shape* of the failure tells the
reader which inductive biases the architecture really depends on.

### Interpretation under the frame story

Mean-pooling absolute frame coefficients is a permutation-invariant
operation on the windowed timesteps.  For row-sMNIST the coordinate
inside a 28-step window carries information (top-of-row vs bottom-of-
row of a digit), but the mean-pool throws much of that away.  We
survive on row because *local* spatial structure between rows is
preserved at the row-of-pixels level: the input to each timestep is a
coherent 28-pixel slice.  Permuted destroys both per-step coherence
and any cross-step locality, leaving mean-pool with bag-of-magnitudes.

## 5. B1: biases × encoder-strategy ablation — what we found

10 seeds × 3 encoder strategies × 3 bias modes for each of MNIST,
Fashion-MNIST, CIFAR-10.  Test accuracies (mean ± SE):

| dataset | bias mode | gaussian | hypersphere | sparse |
|---|---|---|---|---|
| MNIST | data_driven | 0.9708 ± 0.0004 | 0.9708 ± 0.0004 | 0.9708 ± 0.0002 |
| MNIST | random_normal | 0.9714 ± 0.0003 | 0.9609 ± 0.0003 | 0.9719 ± 0.0002 |
| MNIST | zero | 0.9712 ± 0.0002 | 0.9712 ± 0.0002 | 0.9719 ± 0.0003 |
| Fashion | data_driven | 0.8775 ± 0.0006 | 0.8774 ± 0.0006 | 0.8782 ± 0.0004 |
| Fashion | random_normal | 0.8774 ± 0.0004 | 0.8657 ± 0.0005 | 0.8782 ± 0.0003 |
| Fashion | zero | 0.8775 ± 0.0004 | 0.8775 ± 0.0004 | 0.8788 ± 0.0006 |
| CIFAR-10 | data_driven | 0.4972 ± 0.0008 | 0.4972 ± 0.0008 | 0.4985 ± 0.0010 |
| CIFAR-10 | random_normal | 0.4885 ± 0.0006 | 0.4825 ± 0.0008 | 0.4913 ± 0.0006 |
| CIFAR-10 | zero | 0.4834 ± 0.0007 | 0.4833 ± 0.0007 | 0.4853 ± 0.0007 |

Three concrete findings, one mechanism (the abs-hinge geometry of §1):

1. **MNIST and Fashion-MNIST: bias choice does not matter** for the
   data-adapted encoder strategies.  zero ≈ data_driven within ~0.1 pp,
   well inside seed noise.  The data centroid sits far from the
   origin (most pixels black), so a hinge through the origin (zero
   bias) almost never sees `e_i · x` cross zero — abs is effectively
   linear in the data fluctuations, and where you place the hinge
   barely changes the geometry the linear readout sees.
2. **CIFAR-10: data-driven biases buy ~1.4 pp** over zero biases
   (0.4972 vs 0.4834 with hypersphere, 5+ SE separation).  CIFAR-10's
   data centroid sits closer to the origin (the dataset is
   approximately mean ≈ 0.5 per channel after rescaling, but with
   higher variance per pixel), so zero-bias hinges fold through the
   data more often, destroying sign information; data-driven biases
   place each fold point on the manifold and recover what the
   collapse would otherwise lose.
3. **`hypersphere` + `random_normal` is a specific failure mode**:
   on every dataset it lags the corresponding `gaussian` /
   `sparse` rows by ~1 pp.  Mechanism candidate: random Gaussian
   biases on unit-norm encoders place the fold point at random
   distances from the data, frequently in the middle of the data
   distribution where the fold maximally destroys information.
   The other strategies (`gaussian`, `sparse`) have non-unit-norm
   encoders that effectively absorb the bias scale.

Verdict on H1 (zero ≈ data_driven everywhere): **partially supported**.
Holds on MNIST/Fashion, *fails* on CIFAR-10, and the failure has a
clean geometric explanation.  More interestingly, the B7 follow-ups
falsified the original spectral DC reading (which predicted that
*centering* CIFAR-10 should *close* the gap; it actually *opens* it
to +6.3 pp — the largest gap we've seen).  The mechanism is
geometric (where the abs hinge sits relative to data mass), with
"DC offset = constant pixel mean" as one specific case.  See §1.2
"Where the abs hinge sits relative to data" for the post-B7
reframing.

## 5b. B2: Streaming-NEF on sMNIST — what we found

Three panels (10 seeds row, 5 seeds permuted+modes), all under
trace-scaled α=1e-4 (the calibration winner; see §2/§4):

**Row sMNIST** (T=28, K=window).  Test accuracy (mean ± SE):

| n_neurons | accuracy | SE |
|---|---|---|
| 2000 | 97.20% | 0.04% |
| 4000 | 97.97% | 0.02% |
| 8000 | 98.45% | 0.02% |

Doubling n is +0.77 pp (17σ) at 2k→4k and +0.48 pp (12σ) at 4k→8k —
clean monotone scaling.  Reference: published LSTM on row sMNIST is
~98.9% (Bai et al. 2018); we land within ~0.5 pp without recurrence,
without BPTT, with one analytic step per encoder + chunk pass.

**Permuted sMNIST** (T=784).  Test accuracy:

| n_neurons | accuracy | SE |
|---|---|---|
| 2000 | 72.96% | 0.07% |
| 4000 | 75.61% | 0.05% |

Doubling n is +2.65 pp (55σ) — task is far from saturated; with
larger n_neurons we expect the curve continues climbing.  Honest
scope: this is well below the LSTM/TCN range (~90%); the permuted
case removes the spatial locality that the `K`-window prior
captures, exposing the limit of mean-pooling over an unordered bag
of pixels.  This is §6.4 (Streaming Limits) territory and the
bridge to recurrence (Paper III).

**H3 — solve-mode equivalence** (n=4k, α=1e-4):

| solve_mode | accuracy | SE |
|---|---|---|
| batch_acc | 98.03% | 0.02% |
| accumulate | 98.03% | 0.02% |
| woodbury | 98.03% | 0.02% |

Worst pairwise |Δ| over seeds = **0.000 pp** — bit-identical to
reporting precision.  H3 confirmed empirically: the three solve paths
(one-shot, multi-batch accumulator, rank-k Woodbury inverse) reach the
same decoder.  Practical consequence: pick whichever fits the use case
(GPU memory, online vs batch) without accuracy concerns.

**Cautionary baseline** (the α footgun in pictures): the previous
default `trace + α=1e-2` over-regularises across the board:

| n_neurons | trace+1e-2 | trace+1e-4 (winner) | gap |
|---|---|---|---|
| 4000 | 95.89% | 98.03% | +2.14 pp |
| 8000 | 96.92% | 98.44% | +1.52 pp |

And `fixed α=1e-2` is worse still — best at one n (98.07% at 4k) but
blows up at the next (87.27% at 8k, with three of five seeds
collapsed below 80%).  The architectural fix (trace-scaling) is
qualitative, not just quantitative: it stabilises the optimum across
n.

## 6. Encoder strategy ablation (B3)

Seven strategies live in `encoders.ENCODER_STRATEGIES`:

* `hypersphere` — uniform on the unit sphere (universal random frame)
* `gaussian`   — Gaussian directions, non-unit norm (absorbs gain)
* `sparse`     — k-sparse encoders with random ±1 entries (cheap, surprisingly competitive)
* `receptive_field` — random local image patches (spatial locality prior)
* `whitened`   — random direction projected through PCA-whitening (covariance-aware frame)
* `class_contrast` — direction from one class centroid toward the
  nearest other class (label-aware, narrow problems only)
* `local_pca`  — top eigenvector of each neuron's k-NN neighborhood
  (manifold-tangent frame)

Frame-story framing.  Each strategy is a *prior on the frame
directions*.  The universal random frames (`hypersphere`, `gaussian`,
`sparse`) are data-blind but JL-isometric — they preserve manifold
geometry probabilistically with `O(log n / d)` distortion.  The data-
adapted frames (`receptive_field`, `whitened`, `class_contrast`,
`local_pca`) trade universality for *coverage tilted toward the data
manifold* — fewer "wasted" fold directions perpendicular to the
manifold.

Hypothesis to test (H6).  At fixed n_neurons, data-adapted frames
should beat random frames *more* on dataset-with-structure (CIFAR-10,
where image patches mean something) than on dataset-without-low-
dimensional-structure (MNIST, where pixels are nearly independent).

What B1 already tells us about encoders:
* On MNIST and Fashion (low-information-density input), all three
  random strategies sit within 0.1 pp of each other when paired with
  data-driven biases.  Encoder choice is invisible.
* On CIFAR-10 (image structure), `sparse` edges out `gaussian` and
  `hypersphere` by ~0.1 pp under data-driven biases.  Below seed
  noise, but consistent.
* `hypersphere` × `random_normal` biases is a clear failure mode
  (~1 pp drop) — geometric explanation in §1.

What B3 needs to add:
* `receptive_field`, `whitened`, `local_pca` on CIFAR-10 — these are
  the data-adapted strategies that should *outperform* random.
  Hypothesis: 1-3 pp gain over `hypersphere` baseline.
* Sequential / temporal version: do `whitened` or `local_pca` help in
  the streaming setting where the per-window encoder sees a 28- or 1-
  dimensional input rather than 3072-d images?  Open question.
* Combinations.  Mixed frames (half random + half data-adapted) might
  give the best of both.  Not previously tested under multi-seed.

### B3 result (10 seeds × n_neurons=5000, bias_mode=data_driven)

| Strategy | MNIST % | Fashion % | CIFAR-10 % | Verdict |
|---|---|---|---|---|
| hypersphere | 97.08 ± 0.04 | 87.75 ± 0.06 | 49.72 ± 0.08 | baseline |
| gaussian | 97.08 ± 0.04 | 87.75 ± 0.06 | 49.72 ± 0.08 | ≡ hypersphere |
| sparse | 97.08 ± 0.02 | 87.82 ± 0.04 | 49.85 ± 0.10 | ties (everywhere) |
| **receptive_field** | **+0.27 (5.8σ)** | **+0.84 (12.8σ)** | **+5.38 (45σ)** | **clear winner on spatial data** |
| whitened | +0.38 (6.5σ) | +0.59 (8.1σ) | +1.21 (11σ) | small consistent gain |
| local_pca | +0.83 (16σ) | -0.97 (-13σ) | -2.88 (-22σ) | wins MNIST, loses elsewhere |
| class_contrast | -9.36 (-180σ) | -3.14 (-41σ) | -4.44 (-38σ) | systematic failure |

Δ values are paired-by-seed vs `hypersphere`, in pp. `±` numbers are unpaired SE.
Source: `results/sl_clean/b3_strategies_{mnist,fashion,cifar10}_panel_summary.json`.

Story for the paper.
1. **Universal random frames are essentially equivalent.** `hypersphere`,
   `gaussian` (which the encoder builder normalises to the sphere), and
   `sparse` cluster within 0.1 pp on every dataset. The choice between
   them is irrelevant; report whichever is cheapest (sparse, by a hair).
2. **Receptive-field encoders are the headline win.** The +5.4 pp gain
   on CIFAR-10 (49.7 → 55.1%) at fixed n_neurons is by far the largest
   effect we have seen from a frame change. The mechanism is the
   spatial-locality prior: each neuron looks at a small patch of the
   image, so the over-complete frame is a translation-covariant union
   of small frames rather than a single global frame. On MNIST and
   Fashion the gain shrinks (+0.27, +0.84) — pixel structure matters
   less when the digit/garment is roughly centered and stroke-like.
   This is consistent with H6.
3. **Whitened frames give a small consistent edge.** ~+0.4 pp on MNIST,
   +0.6 pp on Fashion, +1.2 pp on CIFAR-10. PCA-aware random directions
   are the cheapest data adaptation, and the gain is monotone in input
   complexity — modest but reproducible.
4. **Heavy data adaptation can hurt.** `local_pca` wins on MNIST
   (+0.83 pp at 97.91%) but *loses* on Fashion (-0.97) and CIFAR-10
   (-2.88). Hypothesis: top-eigenvector-of-kNN frames put too many
   directions parallel to the local manifold, which is exactly where
   the abs-frame loses discrimination — neurons fire only on
   *off-manifold* deviations, and those become rare or noisy on
   richer data. We will report this honestly as a "data adaptation
   isn't free" caveat.
5. **`class_contrast` is a clean failure mode.** -9.4 pp on MNIST,
   -3 to -4 pp elsewhere. The label-conditioned directions point
   toward another class' centroid, which collapses the encoder's
   discriminative capacity onto a 9-direction subspace per anchor.
   We keep this in the paper as a clear "this is what *not* to do"
   counterexample to the temptation of "use labels in the encoder".
6. **Headline default for downstream chapters: `hypersphere` for
   the random-frame baseline, `receptive_field` for spatial data.**
   The "matches MLP without gradients" claim (H2) lives or dies on
   `receptive_field` for CIFAR-10 — at 55.1% it isn't there alone,
   but with ensembling and ConvNEF stacking it is.

Decision for §3 of the paper: present hypersphere as the "vanilla
NEF" baseline, then introduce receptive_field as the only encoder
adaptation that survives every dataset honestly, then note in
limitations that aggressive data adaptation (`local_pca`,
`class_contrast`) can backfire — *coverage of the manifold matters
less than coverage of the manifold's neighborhood*.

Frame-story re-statement.  The B3 outcome refines the hypothesis:
data-adapted frames help only when they *broaden* the set of frame
directions in a translation-covariant way (`receptive_field`,
`whitened`).  When they *narrow* directions onto the manifold itself
(`local_pca`, `class_contrast`), they lose discriminative capacity in
the off-manifold component, which is exactly where abs-coefficient
frames classify.

## 7. Hypothesis ledger (the paper's spine)

H1: data-driven biases ≈ zero biases.  **Partially supported, mechanism reframed (B1+B7).**
    Holds on MNIST/Fashion raw (gap ≤ 0.1 pp); *fails* on CIFAR-10 raw
    by ~1.4 pp.  Original DC-spectral reading falsified by B7
    follow-ups: centering CIFAR-10 *opens* the gap to +6.3 pp (largest
    we've seen); decentering MNIST opens a +1.3 pp gap.  Corrected
    mechanism is geometric — the gap depends on where the abs hinge
    sits relative to the data mass.  Zero bias hinges through origin;
    data-driven bias hinges at training samples.  See §1.2 reframing.
H1b: `hypersphere` encoders with random Gaussian biases are a
     specific failure mode (~1 pp drop on every dataset) — note in
     paper, do not use as default. **(B1)**.
H2: gain range U(0.5, 2.0) ≈ scalar α=1.0.  TBD (B?).
H3: solve modes equivalent given same alpha_eff. **SUPPORTED** (B2,
    worst |Δ| = 0.000 pp across batch_acc / accumulate / woodbury at
    n=4000 with trace-scaled α=1e-4; all three at 98.03 ± 0.02%).
H4: streaming sMNIST-row scales monotonically with n_neurons.
    **SUPPORTED (B2 final panel).** With trace-scaled α=1e-4: 97.20%
    (n=2k) → 97.97% (n=4k) → 98.45% (n=8k); SE ≤ 0.04 pp; +0.77 pp
    (17σ) doubling 2k→4k, +0.48 pp (12σ) doubling 4k→8k.
H4b: regularization must scale with `n_neurons` somehow; fixed α at
     a single hand-tuned value will silently break at out-of-tune n.
     **SUPPORTED** (α-sweep: fixed α=1e-2 best at n=4k = 98.07%, but
     blows up at n=8k = 87.27% ± 6.77 pp with bimodal seed collapse).
H4c: the appropriate α multiplier is *architecture-specific* (one
     scalar for `NEFLayer.fit`, a different scalar for
     `StreamingNEFClassifier`).  **SUPPORTED.** Streaming optimum is
     trace-scaled α=1e-4, two orders of magnitude below the
     `NEFLayer.fit` 1e-2 default; reflects the much smaller trace per
     neuron after T-step mean-pooling.
H5: streaming sMNIST-permuted is a published limit of delay-line +
    abs + mean-pool.  Mitigations queued (window, position encoding,
    pool variants, data-adapted encoders); we expect them to fail or
    to converge to the unordered-bag-of-pixels baseline.
H6: encoder-strategy ranking matches frame-prior intuition (B3).
    Specifically: data-adapted frames beat random frames on
    structured inputs (CIFAR-10) but not on unstructured ones (MNIST).
H7 (DC-fold prediction): per-channel centering CIFAR-10 closes the
    data-driven-vs-zero bias gap; DC-shifting MNIST opens one.
    **PARTIALLY SUPPORTED — mechanism reframed.** B7 results
    (10-seed × 9-cell each) refute the "remove DC ⇒ no gain" half
    but confirm the "add DC ⇒ gain appears" half:

    | scenario               | DD%   | zero%  | gap     | predicted |
    |------------------------|-------|--------|---------|-----------|
    | MNIST raw (B1)         | 97.08 |  ~97.0 |   0.0   |  ~0       |
    | **MNIST decentered**   | 97.12 |  95.78 | **+1.3**|  positive |
    | CIFAR-10 raw (B1)      | 57.79 |  56.40 |  +1.4   |  ~+1      |
    | **CIFAR-10 centered**  | 49.72 |  43.41 | **+6.3**|  ~0       |

    The corrected mechanism is geometric, not spectral: **data-driven
    biases place each abs hinge at a training sample**; zero biases
    place every hinge at the origin. The gap depends on how far the
    data centroid sits from the origin.

    - MNIST raw: data clusters near 0 (mostly black pixels) ⇒ zero
      bias *is* near data ⇒ no gain.
    - MNIST decentered (x + 0.5): data shifted ~0.5 from origin
      ⇒ zero biases land in nearly-empty space ⇒ DD recovers.
    - CIFAR-10 raw: mean ~0.5; zero bias not great but workable.
    - CIFAR-10 centered: data centroid AT origin ⇒ zero bias puts
      every abs hinge symmetrically through data mass — the worst
      case ⇒ DD opens a 6 pp gap.

    Geometric reading via the frame interpretation: with abs
    activation a_i = |α_i (e_i · x) + b_i|, zero bias makes the
    hinge a hyperplane through the origin; if the data is centered
    at the origin half its mass falls on each side and the
    activation modulus loses sign information symmetrically. Setting
    b_i = -α_i (e_i · d) places the hinge at training sample d,
    so the activation is "small near a known datum, growing as we
    move away" — the natural reference frame for over-complete
    decoding. See B7 panels in `results/sl_clean/biases_*_panel.json`.

H8 (linear-collapse): a linear-activation NEFLayer at n_neurons ≫
    d_in scores no better than 784→10 ridge regression. **SUPPORTED.**
    Linear NEFLayer at n=1000 and n=4000 both reach 86.0–86.2% on
    MNIST (within 1 pp of plain ridge at 85.3%); abs NEFLayer at
    n=5000 reaches 97.1%.  Δ ≈ 11 pp is the entire contribution of
    the nonlinearity.  See `results/sl_clean/h8_linear_collapse.json`
    and `benchmarks/sl_clean/h8_linear_collapse_check.py`.

## 8. Things we owe the reader

- A figure that places NEF-style solves in the broader analytic-readout
  family (random feature models, ELMs, reservoir computing, kernel ridge
  regression).  Frame coefficients are the bridge.
- A clear statement of *what we are not claiming* — we are not claiming
  to beat SOTA on MNIST or CIFAR-10.  We are presenting a coherent
  framework with honest scope.
- A reproducibility appendix: `code_identity` content addressing,
  `set_sl_seed` discipline, dataset hashes, hardware fingerprint.
- A short note on the engineering improvements (Cholesky over LU, float64
  running stats, max_tokens encoding, chunk_stats pooling) — not the
  centre of the paper but worth a paragraph because they are what make
  multi-seed studies tractable on commodity CPUs.
- A "negative results" section.  We tried these and they did not work
  / matched the random baseline / did not beat the simpler thing.  This
  is where permuted-sMNIST mitigation attempts live, and where any
  "data-adapted encoder didn't help on dataset X" lives.  The paper is
  *more* trustworthy with a negative-results section than without.
- A clear statement of where biases come from in the frame-story
  framing.  Bias `b_i = -α_i (e_i · μ_i)` where `μ_i` is the per-neuron
  reference point (a training sample under data-driven, the origin
  under zero, a random vector under random_normal).  This makes
  biases a *placement* of the fold point in input space, not a free
  parameter — and clarifies why "zero biases" can be the right default
  for centered data: the fold is already in a sensible place.
- A short discussion of the relation to the original NEF.  We use NEF's
  encode-decode skeleton but our activation conventions and our refusal
  to use spiking/temporal coding put us closer to random feature models
  than to neuromorphic computing.  Honest framing prevents both
  audiences (NEF and ML) from being confused by the other's vocabulary.
- The formal universality claim with a citation, not just hand-waving.
  Cybenko/Hornik for sigmoid; ReLU universality is e.g. Yarotsky.  Abs
  is a piecewise-linear approximator; the universality follows from the
  ReLU case via `|x| = ReLU(x) + ReLU(-x)`.  Worth one careful footnote.


## 9. What is novel here, what is replication, what is honest framing

Internal standard: no claim-pumping and no hiding failures.  The paper
should say plainly:

**Replication, not novelty.**
- Single-layer NEFLayer.fit() on MNIST is roughly an Extreme Learning
  Machine (Huang et al. 2006).  Our solver is the same Tikhonov ridge
  regression on fixed features.  We will cite ELM and acknowledge this
  honestly.  Our contribution at the single-layer level is *the framing
  and the engineering* (frame-coefficient interpretation, content-
  addressed reproducibility, multi-seed discipline), not a new family
  of feature extractors.
- ConvNEF on CIFAR-10 is in the same family as Coates et al. 2011
  (k-means features) and Saxe et al. 2011 (random weights are good
  enough).  We replicate, organize, and re-frame; we do not surpass.
- Random-feature, kernel-ridge, reservoir-computing literature has
  most of the theoretical machinery already.  We will cite Rahimi &
  Recht (2007), Jaeger (2001), Maass et al. (2002), and connect back
  to them rather than pretend originality.

**What we think *is* novel and worth presenting:**
- The frame-coefficient framing is, as far as we know, not the
  standard way of explaining NEF.  It cleanly explains why abs/ReLU
  are sufficient (§0), why fold placement matters (§1.2), and why
  data-driven biases pay off in some regimes but not others (§5).
  This is a pedagogical
  contribution; it lets practitioners reason about the geometry
  rather than treat the feature map as a black box.
- Streaming-NEF (delay-line + abs + mean-pool + analytic readout)
  on sMNIST-row.  Earlier internal results showed ~98.6 % at n=10000
  matching gradient-trained LSTMs while training in seconds rather
  than minutes.  We re-validate under multi-seed and report whatever
  comes out without inflation; the numerical comparison to LSTMs is
  the published-style empirical contribution.
- The trace-scaled-α regularization story (§2 + §2b).  Not a new
  algorithm — `solvers.tikhonov` already does this — but the
  cautionary baseline and the architecture-specific α multiplier
  story is, we believe, not in the literature in this exact form.
  Useful for practitioners who follow our recipe in different
  domains.
- A reproducibility infrastructure (rl_clean style) that makes
  multi-seed analytic-readout studies cheap and *resumable*.  Not
  paper-novel; lab-novel.

**Calibrated uncertainty (deferred but worth mentioning).**
The Tikhonov posterior covariance `(AᵀA + reg I)⁻¹` is a closed-form
Gaussian uncertainty over decoders.  We do not exploit this in
Paper I but the infrastructure is already there.  One sentence in
the discussion section, pointing to a future paper, is enough.

## 10. Operational notes (delete before submission)

These are reminders to ourselves; they get cut from the published
text.

- Commit cadence: never commit while a sweep that depends on
  `code_identity` is running.  The watched paths are
  `("src", "benchmarks", "tests", "pyproject.toml")`; `docs/` is safe.
- Always include `flush=True` or `python -u` in long-running scripts;
  `multiprocessing.Pool` workers buffer stdout silently otherwise.
- Never pipe long-running progress through `head` — broken pipe
  silently kills `print()` calls in workers.
- Default RNG hygiene: do *not* rely on the global PyTorch RNG; pass
  `torch.Generator()` explicitly.  We have been bitten before.
- When designing sweeps, sample at least one combination at each
  expected boundary of the regime (e.g. `n ∈ {2k, 4k, 8k}` not just
  `{4k, 8k}`) so we can fit a curve, not just compare two points.
- Honest-by-default for the paper.  Every claim should be reproducible
  from the JSON in `results/sl_clean/`.  A claim without a JSON is
  not a claim.

## 11. NEF vs NEF-inspired — naming decision

Eliasmith and Anderson's *Neural Engineering Framework*
(`Neural Engineering: Computation, Representation, and Dynamics in
Neurobiological Systems`, MIT Press 2003) frames its three principles
around **spiking neuron models**: representation by population spike
rates, transformations decoded from those rates, and dynamics expressed
through recurrent synaptic projections.  The signature use case is
neural-control of continuous-time dynamics with leaky-integrate-and-fire
populations; the optimization step is least-squares decoder fitting,
identical to ours.

What we keep verbatim from NEF:
- fixed encoders (`e_i`), per-neuron gain (`α_i`), bias as a function
  of an intercept (`b_i = -α_i (d_i · e_i)` in our data-driven case);
- analytic least-squares decoders against an arbitrary target function
  of the encoded variable;
- the conceptual split: representation (encode), transformation
  (decode), and the freedom to compose decoded signals.

What we change:
- **Rate neurons with `abs` activation** rather than spiking LIF.  This
  is more than a numerical convenience: the abs-rate readout is what
  gives us the frame-coefficient interpretation §1, and it makes the
  decoder's job a linear inverse problem with a clean condition-number
  story.
- **Data-driven biases** (`centers = x_train`) rather than the random
  intercept distributions used in the textbook.  This places each
  neuron at a training sample, making the resulting representation
  closer to a learned dictionary than to a generic frame.
- **Sufficient-statistic-first programming model**: `AᵀA`, `AᵀY` as
  the universe, supporting incremental, online, ensemble, and
  per-action solves with no extra machinery.  This is implicit in
  NEF (the least-squares decoder pre-supposes it) but we make it the
  central object.
- **No dynamics step.**  We do not use NEF's principle 3 (recurrent
  dynamics through decoded projections) anywhere in this paper.  The
  recurrent and streaming variants reuse encoders and decoders with
  pooling, not recurrent dynamics solve.

**Decision for the paper.** We continue to call the method "NEF" with
a clearly stated deviation: the opening of §1 will (a) attribute the
core encode-activate-decode pattern to Eliasmith & Anderson and cite
their book; (b) state explicitly that we use rate-`abs` neurons, that
our biases are data-driven, and that we do not use the dynamics
principle; (c) name our specialization "leenef" (or, in mathematical
prose, "the analytic-rate variant of NEF") so any later reader who
looks for spiking-NEF behavior does not feel misled.  This is the
same way other communities cite NEF (Nengo, BCM, etc.) — as a starting
point that is then specialized.

If a later reviewer pushes back, "NEF-inspired" / "NEFi" is the
fallback rename, but we expect the explicit deviations paragraph to
satisfy good-faith readers.

The trilogy's shared opener (§0pre — the six principles) is *our*
distillation, not Eliasmith & Anderson's; the paper should make this
distinction clear by attributing each principle to its origin.

## 12. Sparse-reward RL — relevant background for Paper I framing

The sparse-reward research line (worktrees `leenef-sparse-reward-c`
and `leenef-sparse-reward-g`; primary doc
[`../leenef-sparse-reward-c/docs/rl_sparse_notes.md`](../leenef-sparse-reward-c/docs/rl_sparse_notes.md))
is canonical reading for Paper III but worth mentioning here because
its §1.2 ("the leenef essence") is the *clearest existing spelling-out
of the base principles* anywhere in the project — it predates and
informed §0pre above.  Differences worth tracking when we synthesize
Paper III:

- Sparse-reward §1.2 enumerates **six principles** in the same shape
  as our §0pre, with one extra emphasis: principle 6
  (per-class/per-action additivity) is what makes CL and per-action
  FQI work.  Paper II inherits this and Paper III rests on it.
- The mechanism stack — D1 (NEF Successor Features), D2 (leverage
  novelty), D4 (hindsight + `eval_goal` projection) — is each just
  another reuse of `AᵀA / AᵀY` over a different target.  Principle 4
  ("targets are where task structure enters") does most of the work.
- Multi-seed evidence (3 seeds) for MountainCar and Acrobot under
  sparse reward; explicit clean negatives (LunarLander regression,
  toy-chain `eval_goal` failure) recorded in the same notes.

Paper I should reference this line only briefly (in §1, when we
introduce the sufficient-statistic framing, we can foreshadow that
the same primitives carry CL and RL — with citations to the live
sparse-reward branches and to Paper II/III for the specifics).

## 13. Configuration provenance and code-identity policy (operational)

Our `code_identity` (40-hex SHA over
`src benchmarks tests pyproject.toml`) discriminates result rows
across library changes; the policy is documented in
[`docs/sl_clean_sweeps.md`](sl_clean_sweeps.md) under
"Code-identity timeline".  Headline reading for Paper I:

- Six panels are pinned to six different identities; we never mix
  identities inside an aggregate; per-row reproducibility is exact.
- The fixes between identities (Cholesky solve, clone-fix, alpha
  guard, float64 accumulators) only touch `continuous_fit` /
  Woodbury paths — they do not change `NEFLayer.fit()`'s analytical
  decoder solve, which is the path on which B1/B3/B7's headline H1
  and H6 numbers rest.
- B2 (streaming) lives on the post-Cholesky/clone-fix identity, so
  the H3/H4 streaming numbers are taken under the now-correct
  numerics.  No re-runs are required.

When the paper text is written, every numeric claim should cite the
panel JSON it draws from, and the code_identity for that panel can
be read directly from the JSON if a reviewer asks.

## 14. B4 verdict — gradient-trained MLP vs single-layer NEF (H2)

Final 5-seed numbers (panel summaries
`results/sl_clean/b4_mlp_{mnist,fashion,cifar10}_panel_summary.json`):

| dataset | NEF (hypersphere n=5000) | tuned MLP best | Δ (MLP − NEF) |
|---|---|---|---|
| MNIST | 97.08 ± 0.04 | **98.10 ± 0.08**  (407k params, 22 ep, 58 s) | +1.02 pp |
| Fashion-MNIST | 87.75 ± 0.06 | **89.39 ± 0.08** (1.86M params, 24 ep, 254 s) | +1.64 pp |
| CIFAR-10 | 49.72 ± 0.08 | **53.73 ± 0.22** (4.21M params, 34 ep, 327 s) | +4.01 pp |

The honest single-line story: **a tuned MLP beats hypersphere NEF, and
the gap grows with task difficulty.**  The interesting twist is that
B3's `receptive_field` NEF reaches **55.10 ± 0.10** on CIFAR-10, which
beats the best B4 MLP by +1.37 pp at a fraction of the wall time.
The paper's framing (consistent with §0pre principle 1) is that the
encoder is where representational work happens; the decoder is just
the analytic readout.  A good encoder closes the gap; a generic
encoder doesn't.

Things to write carefully and not over-claim:
- We are not claiming NEF beats deep convnets.  We are showing that a
  single linear analytic readout on top of a fixed feature extractor
  can match or beat a tuned gradient-trained MLP on these benchmarks.
- We are not claiming MLPs are obsolete.  MLPs win the generic
  comparison; NEF wins when the encoder carries a useful prior.
- The wall-time comparison should be reported with hardware
  qualified (CPU, single thread for both is the default).  The MLPs
  we ran are CPU baselines; the headline timing comparison is also
  CPU-only.  GPU numbers would be a separate exercise.
