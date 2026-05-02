# Sufficient Statistics as the Universal Primitive

## Supervised Learning via the Neural Engineering Framework

> **Status: draft in progress.**  This is the working manuscript for
> Paper I of the leenef foundational trilogy.  Section structure
> tracks `paper_i_outline.md`; numerical claims are sourced from
> `results/sl_clean/` JSON files and remain in lockstep with
> `paper_insights.md`.  Cited Eliasmith & Anderson chapters are
> from *Neural Engineering: Computation, Representation, and
> Dynamics in Neurobiological Systems* (MIT Press, 2003) unless
> stated otherwise.

---

## Abstract

> **(stub — to be written last, after §6 numbers stabilise)**
> Single regularised least-squares solve over fixed-encoder
> activities, four design choices that make it work, and a streaming
> classifier that turns the same primitive into competitive
> sequence classification without recurrence.

---

## §1 — The six principles of leenef

We build on the *Neural Engineering Framework* of Eliasmith and
Anderson (2003): a population of neurons with fixed encoders
represents a vector quantity, and any function of that vector is
computed by a linear decoder fitted to the population's activity.
Their book frames this around three principles — representation,
transformation, and dynamics — and the canonical implementation
uses spiking leaky-integrate-and-fire neurons whose firing rate
encodes the represented quantity.  We keep the encoder–decoder
split, the analytic least-squares fit, and the conceptual freedom
to compose decoded signals; we depart in three places.

1. We use **rate neurons with `abs` activation**, not spiking LIF.
   `abs` is the smallest pointwise nonlinearity that escapes the
   linear-collapse argument of §2; it is also what gives our
   activations the frame-coefficient interpretation that the rest
   of the paper turns on.
2. We use **data-driven biases**.  The bias of neuron *i* is
   `b_i = -α_i (e_i · d_i)` for a sample `d_i` drawn from the
   training set, placing the *i*-th `abs` fold on the data
   manifold rather than on a generic random intercept.
3. We do not use NEF's **dynamics principle** anywhere in this
   paper.  Streaming and recurrent variants reuse encoders and
   decoders with pooling, not a recurrent dynamics solve.

When this paper says "NEF" it means this *analytic-rate-`abs`
specialisation*; when distinction matters we call our local
specialisation **leenef**.  Readers who arrive expecting Nengo-
style spiking populations should read NEF here as "the
encode–decode pattern of Eliasmith & Anderson, with rate-`abs`
neurons and data-driven biases."

The trilogy's shared opener is six operational claims, distilled
from the program's running code (`streaming.py`, `layers.py`,
`recurrent.py`, `rl.py`) and the sparse-reward research line
(see Paper III).  These are *our* distillation; principles 1–4
inherit from NEF, principles 5–6 are leenef-specific.

1. **Encoders are fixed** (random or data-adapted).  Decoders are
   the only thing we ever fit.
2. **Decoders fall out of one analytic step:**
   `D = (AᵀA + αI)⁻¹ AᵀY`.  No learning rate, no schedule, no
   gradient flow between layers.
3. **`AᵀA` and `AᵀY` are the universe.**  Continual learning,
   streaming over time, ensembles, online updates, per-action
   FQI — all of them are different ways of *additively combining*
   sufficient statistics.
4. **Targets `Y` are the only place where task structure enters.**
   The encoder does not care what the task is; the decoder column
   does.
5. **Time is collapsed by pooling, not by recurrence.**  The
   streaming classifier mean-pools delay-line activities into a
   fixed-size representation; no BPTT, no recurrent solve.
6. **Per-class additivity is what makes continual learning work.**
   Each class owns its own decoder column, so accumulating new
   task statistics never interferes with old columns.  (Forward-
   look — owned by Paper II; we exhibit the additivity property
   here without claiming it solves CL.)

The rest of Paper I is one of these primitives applied somewhere
new: §3 (method) is principles 1–4, §6 (streaming classifier) is
principle 5, and the multi-layer experiments (§5.5) are the
load-bearing negative result that says principle 2 does not
extend through stacked nonlinearities — which is why we stay
single-layer for the rest of the trilogy.

---

## §2 — Why over-representation needs nonlinearity

A linear activation collapses the entire scheme.  If
`a_i = α_i (e_i · x)`, then for any decoder `D ∈ ℝ^{n × d_out}`,

```
y = Dᵀa = Dᵀ diag(α) E x = (Dᵀ diag(α) E) x = W x
```

is a single linear map, regardless of `n`.  Over-completeness
buys **nothing**: a 100 000-neuron linear network is exactly as
expressive as a single linear projection of input onto output.
The frame `{α_i e_i}` is just an absurdly redundant
parametrisation of a `d_in × d_out` matrix.

We have verified this in a sanity-check experiment (sweep H8 in
`docs/sl_clean_sweeps.md`): a linear-activation NEFLayer on
MNIST plateaus at the linear-regression accuracy of `Wx`,
unchanged across `n ∈ {500, 5000, 50 000}` to within seed noise.
The collapse is empirical, not just algebraic.

The minimal repair is a pointwise nonlinearity that is *not*
representable as a linear scaling.  `abs` is the smallest such
repair we can imagine:

```
a_i = |α_i · e_i · x|
```

Geometrically, `|e_i · x|` is the **unsigned distance of x from
the hyperplane `{e_i · z = 0}`**.  Each neuron measures one such
unsigned distance.  A linear readout over `n` unsigned-distance
measurements is a sum of folded half-planes — a piecewise-linear
function with up to `n` fold lines.  With `n` large and `e_i`
covering directions densely enough, this function class is
universal on compact subsets of ℝ^d.  ReLU is the same idea
after a half-fold (drop the sign instead of folding about the
origin); `|x| = ReLU(x) + ReLU(-x)`, so an `abs` neuron is two
mirrored ReLU neurons whose spans coincide once the decoder is
solved.

### The frame-coefficient interpretation

Set `u_i = α_i e_i`.  Then
`a_i = |u_i · x|` is the magnitude of the *i*-th frame
coefficient of `x` in the (over-complete, possibly random) frame
`{u_i}`.  When biases are non-zero, the frame becomes
*manifold-local*:

```
a_i = |α_i · e_i · (x − d_i)|        (data-driven bias case)
```

Each neuron then asks "how far is x from training sample `d_i`,
along direction `e_i`?"  The decoder is the linear functional
that reads off the target from these magnitudes.  This framing
will recur throughout the paper:

- **Encoder strategies** (§5.2) are different priors over the
  frame directions `{u_i}`: hypersphere = isotropic; whitened =
  data-covariance-aware; receptive_field = spatial-locality-aware.
  Random, near-isotropic frames are universal but data-blind;
  data-adapted frames trade universality for efficiency.
- **Per-neuron gain `α_i`** (§5.3) rescales each frame element,
  equivalent to non-uniform measurement of `x`.  The default
  `U(0.5, 2.0)` range gives mild adaptive scaling that empirically
  beats `α ≡ 1`.
- **Bias placement** (§5.1) selects where each `abs` fold sits
  in input space.  The next paragraph shows why this matters
  more than it might appear.

### Why bias placement matters: the manifold view

Real data lives on (or near) a low-dimensional manifold inside
the ambient input space.  Two complementary roles emerge.

- **Random encoders** approximate a near-isometric measurement
  of that manifold (Johnson–Lindenstrauss / restricted-isometry
  style).  Enough random directions in high dimension preserve
  pairwise distances and angles, so the data manifold's geometry
  is carried faithfully into the *n*-neuron frame coefficients.
- **Data-driven biases** place each neuron's measurement origin
  *on* the manifold itself.  The global frame becomes a
  *manifold-aware local* frame: tangent-style coordinates from
  many neighbourhoods at once.

Whether local framing actually helps over global (zero-bias)
framing depends on where the `abs` fold sits relative to the
data:

- If the data centroid is *at* the origin, zero bias is a worst
  case: half the data mass falls on each side of every fold and
  `abs` collapses sign symmetrically.
- If the data centroid is *far from* the origin, zero bias is
  accidentally sane (sign rarely flips) and `abs` is effectively
  linear in the data fluctuations.

This single geometric axis predicts both halves of the §5.1
ablation table; we resisted reading it spectrally (as a
DC-component story) once the centred-CIFAR-10 follow-up (§5.1)
falsified the spectral prediction.

---

> **(continues — §3 method, §4 α-scaling, §5 single-layer
> ablations, §6 streaming-NEF, §6b MLP comparator, §7 honest
> framing, §8 forward-look)**
