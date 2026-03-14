# Analytical Target Propagation for NEF Networks

## Motivation

Our hybrid training strategy uses backpropagation through the full network
to update encoder weights.  This works — hybrid reaches 98.6% on MNIST —
but the gradient signal is global: every encoder update depends on loss
information that flows backward through all layers.  This is expensive for
deep networks, biologically implausible, and arguably un-NEF-like.

**Key observation:** NEF already gives us analytical decoders at every
layer.  Target propagation needs learned "inverse models" to propagate
targets backward — and our decoders *are* inverse models.  We solve them
for free.

## Background

### Target Propagation (Lee et al., 2015)

Standard target propagation replaces backprop with layer-local targets:

1. Forward pass: compute all layer activities.
2. Compute a **target** for each layer's activities — what they *should*
   have been to reduce the loss.
3. Update each layer's weights to match its local target.

The target for layer *l* is computed by inverting the forward mapping of
layer *l+1*:

```
target_l = g_{l+1}(target_{l+1})
```

where `g_{l+1}` is a learned decoder (approximate inverse) of the
forward function `f_{l+1}`.

**Difference Target Propagation (DTP)** adds a correction term to prevent
error accumulation:

```
target_l = a_l + g_{l+1}(target_{l+1}) − g_{l+1}(a_{l+1})
```

This ensures that when `target_{l+1} = a_{l+1}` (no change needed), the
target for layer *l* is exactly its current activities.

### NEF Representational Decoders

In Eliasmith's NEF, every neural population has a **representational
decoder** that recovers the encoded quantity from the population's
activities:

```
x̂ = a @ D_repr    where D_repr = argmin_D ||A @ D − X||²
```

This is solved analytically via regularised least-squares — the same
solver we already use for output decoders.

The representational decoder is the inverse model that target propagation
needs.

## Algorithm: NEF-TP (Analytical Target Propagation)

### Architecture

Each layer *l* in the network has:
- **Encoders** `E_l` — project input to neuron space (trainable)
- **Bias** `b_l` — per-neuron offset (trainable)
- **Gain** `g_l` — per-neuron scaling (fixed)
- **Activation** — nonlinearity (default: abs)

The output layer additionally has a **task decoder** `D_out` (solved
analytically).  During TP training, every layer gets a **representational
decoder** `D_repr_l` (also solved analytically, recomputed each iteration).

### Training Loop

For a network with L hidden layers and one output layer:

```
for each iteration:
    # ── Forward pass ──────────────────────────────────────
    a[0] = x                                    # input
    for l = 1 to L:
        a[l] = activate(gain_l * (a[l-1] @ E_l.T) + b_l)  # hidden

    a[L+1] = activate(gain_out * (a[L] @ E_out.T) + b_out)  # output neurons

    # ── Solve decoders (all analytical) ───────────────────
    D_out = solve(a[L+1], targets)              # task decoder
    for l = 1 to L+1:
        D_repr[l] = solve(a[l], a[l-1])         # representational decoders

    # ── Compute targets (backward, no backprop) ──────────
    error = a[L+1] @ D_out − targets
    target[L+1] = a[L+1] − η · error @ D_out.T  # output target

    for l = L down to 1:
        # DTP correction: exact when target = current activities
        target[l] = a[l] + (target[l+1] − a[l+1]) @ D_repr[l+1]

    # ── Local encoder updates (independent per layer) ────
    # These can be parallelised — no dependencies between layers!
    for l = 1 to L+1:                           # in parallel
        loss_l = ||encode_l(a[l-1]) − target[l]||²
        update E_l, b_l with ∇loss_l            # single-layer gradient
```

### Why This Works

1. **D_repr is the NEF representational decoder** — it's what the
   population "thinks" the input was.  This is exactly the inverse model
   that target propagation needs.

2. **All decoders are solved analytically** — no gradient descent for
   decoder weights, no extra training loops.

3. **Encoder updates are fully local** — each layer only needs its own
   input, its current activities, and its local target.

4. **DTP correction prevents error accumulation** — the difference
   `(target[l+1] − a[l+1]) @ D_repr[l+1]` ensures targets degrade
   gracefully across layers.

### Step Size η

The output target `target[L+1] = a[L+1] − η · error @ D_out.T` pushes
activities in the direction that reduces output loss.

- `η` too large → target far from feasible activity space → noisy encoder updates
- `η` too small → target ≈ current activities → no learning signal
- Default: `η = 0.1`, tunable

An alternative: **normalised step**, scaling by `||D_out||²` so that
`η = 1` corresponds to a Newton-like step in activity space.

## Comparison with Hybrid Training

| Property              | Hybrid               | NEF-TP                |
|-----------------------|----------------------|-----------------------|
| Gradient scope        | Full backprop        | Single layer          |
| Decoder solves / iter | 1 (output only)      | L+1 (all layers)     |
| Encoder gradient cost | O(L × forward)       | O(1 × forward) each  |
| Parallelisable        | No (chain rule)      | Yes (layer-independent) |
| Memory                | Full computation graph | Per-layer only        |
| Biological plausibility | Low                | Higher (local rules)  |

**Expected trade-off:** TP solves more decoders per iteration (cheap with
our solvers) but saves on gradient computation.  For deep networks, TP
should be faster per iteration.  Whether it converges in fewer or more
iterations is the key experimental question.

## Activation Considerations

**abs (default):** Many-to-one — `abs(z) = abs(-z)`.  Two inputs
differing only in sign along an encoder direction produce the same
activity.  The representational decoder cannot distinguish them.  In
practice, with enough random encoder directions and data-driven biases,
this is unlikely to be a problem: the full activity vector is still
approximately injective (Johnson-Lindenstrauss argument).

**relu:** One-to-one on positive half-space, zero on negative.  Loses
information about negative projections but does not conflate positive and
negative.  The representational decoder has a harder time (zero activities
carry no information) but the non-zero activities are fully informative.

**Prediction:** abs should work well for TP since the representational
decoder has richer (always-nonzero) activities to work with.  ReLU might
struggle with sparse activities in deeper layers.  We test both.

## Related Ideas (for future exploration)

### 1. Feedback Alignment (Lillicrap et al., 2016)

Replace `D_out.T` in the target computation with a random fixed matrix.
Our encoders are already random, creating a natural symmetry.  But
analytical decoders are strictly better — they're "free" to compute and
give a proper inverse.

### 2. HSIC Bottleneck (Ma et al., 2020)

Train each layer to maximise the Hilbert-Schmidt Independence Criterion
between its representation and the target, independently.  The kernel
matrix is computed from the activity matrix, and the optimum has a
closed-form related to our normal-equations solver.  This is fully local
and could complement TP.

### 3. Forward-Forward Algorithm (Hinton, 2022)

Each layer learns to distinguish "positive" data (real) from "negative"
data (corrupted), using only local information.  No backprop needed.
Could be combined with our analytical decoders: solve D after each layer
learns good encodings via forward-forward.

### 4. Predictive Coding Networks (Rao & Ballard, 1999; Millidge et al., 2022)

Hierarchical predictive processing where each level predicts the activity
of the level below, and only prediction errors propagate upward.  Our
representational decoders serve as the generative model (top-down
predictions).  The prediction error at each layer could drive encoder
updates — conceptually equivalent to our TP approach but with an
iterative inference phase that relaxes to equilibrium before weight
updates.

### 5. Efficient Target Propagation (Bao et al., AAAI 2024)

Independent discovery of analytical decoder solutions for target
propagation.  They derive closed-form feedback weights from a
Jacobian-matching loss.  Our NEF decoders achieve the same effect from
neuroscience principles.

### 6. Fixed-Weight DTP (Shibuya et al., 2023)

Shows that even fixed (untrained) feedback weights work for DTP if the
forward weights are well-conditioned.  This is encouraging for our setup:
if the initial random representational decoders are already good enough,
the first iteration of TP may be effective even before iterative
refinement.

## References

- D.-H. Lee, S. Zhang, A. Fischer, Y. Bengio, "Difference Target
  Propagation", *ECML-PKDD*, 2015.
- T. P. Lillicrap, D. Cownden, D. B. Tweed, C. J. Akerman, "Random
  synaptic feedback weights support error backpropagation for deep
  learning", *Nature Communications* 7, 2016.
- R. P. N. Rao & D. H. Ballard, "Predictive coding in the visual
  cortex: a functional interpretation of some extra-classical
  receptive-field effects", *Nature Neuroscience* 2, 1999.
- B. Millidge, A. Tschantz, C. L. Buckley, "Predictive Coding
  Approximates Backprop along Arbitrary Computation Graphs", *Neural
  Computation* 34(6), 2022.
- Y. Bao et al., "Efficient Target Propagation by Deriving Analytical
  Solution", *AAAI*, 2024.
- T. Shibuya et al., "Fixed-Weight Difference Target Propagation",
  *AAAI*, 2023.
- G. Hinton, "The Forward-Forward Algorithm: Some Preliminary
  Investigations", *arXiv:2212.13345*, 2022.
