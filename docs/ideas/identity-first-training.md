# Identity-first training for multi-layer NEF

Status: **idea — not yet implemented**

## Motivation

Eliasmith and Anderson's NEF distinguishes *representation* from
*transformation*.  A population's activities are its **representation** of
the input.  A "representational decoder" reconstructs the input from those
activities; a "transformational decoder" extracts some function f(x)
instead.  Crucially, the same representation supports both — the
transformation lives entirely in the decoder.

> "Rather than decoding a representation of x from a spike train, we are
> decoding a function of x — that is, f(x) — from that same spike train."
> — Eliasmith, *How to Build a Brain*, §2.2.2

Our multi-layer architecture already reflects this: hidden layers
`encode()` only (creating representations), and the output layer decodes
(performing the transformation).  But currently, encoder training
(hybrid/E2E) optimises directly for the classification target.  The idea
here is to first train encoders for **faithful representation** (identity),
then solve the transformation analytically in the output decoders.

## Algorithm

### Phase 1 — Representation learning (identity pre-training)

Train all encoders so that the network's hidden activities can reconstruct
the input x.  This is hybrid-style training but with identity as the
target:

```
for each iteration:
    h = hidden_encode(x)                  # activities, no grad
    D_id = solve(h, x)                    # temporary identity decoders
    h = hidden_encode(x)                  # activities, with grad
    loss = mse(h @ D_id, x)              # reconstruction error
    loss.backward()                       # gradients to hidden encoders
    optimizer.step()
```

The identity decoders D_id are never stored in the model — they exist only
to provide the gradient signal.  Their shape is [n_last_hidden, d_in],
e.g. [1000, 784] for MNIST.

### Phase 2 — Transformation (analytical decode)

Freeze all encoder parameters.  Pass x through the full network (hidden
encoding → output encoding), solve output decoders for the actual target:

```
h = hidden_encode(x)         # trained hidden representations
a_o = output.encode(h)       # output layer activities
output.decoders = solve(a_o, targets)
```

One analytical solve — no gradients needed.

### Optional Phase 3 — Hybrid refinement

Run a few hybrid iterations with the actual target to shift encoders from
general (information-preserving) to discriminative (task-specific)
representations, then re-solve output decoders.

### Optional Phase 4 — E2E refinement

Like hybrid→E2E, run SGD on all parameters from the warm start.

## Variants

### Variant A: Hidden-only identity pre-training

Only hidden encoders participate in identity training.  The output layer
is untouched during Phase 1 — its encoders remain random.  After Phase 1,
output decoders are solved for the target through the random output
encoding.

**Pro:** Simplest implementation.  
**Con:** Output layer's random encoders may bottleneck the trained hidden
representations.

### Variant B: Full-network identity pre-training

Train all encoders (hidden + output) during Phase 1.  Use temporary
decoders from the output layer's activities to x:

```
h = hidden_encode(x)                      # with grad
a_o = output.encode(h)                    # with grad
D_temp = solve(a_o.detach(), x)           # temp decoders [n_out, d_in]
loss = mse(a_o @ D_temp, x)
loss.backward()                           # gradients to ALL encoders
```

The output layer's decoders change shape between Phase 1 ([n_out, d_in])
and Phase 2 ([n_out, n_classes]), but only D_temp is used during Phase 1
and it's never stored in the model.

**Pro:** Output encoders also learn good representations.  
**Con:** More expensive per iteration (gradient flows through more
parameters).

## Theoretical analysis

### Why it might help

1. **Smoother loss landscape.**  Identity reconstruction is a well-
   conditioned problem — every encoder direction captures *some*
   information about x.  Classification is noisy (53% accuracy on CIFAR-10
   means lots of incorrect gradient signals).

2. **NEF-principled.**  Directly implements Eliasmith and Anderson's
   separation: learn good representations first, then decode any desired
   transformation.

3. **Better initialisation region.**  Identity-trained encoders may sit in
   a region of parameter space from which hybrid refinement converges to
   better optima.

### Why it might not help

1. **Reconstructive vs discriminative.**  For classification, you don't
   need to represent ALL of x faithfully — you need features that separate
   classes.  Identity training spreads encoder capacity across all input
   dimensions, including uninformative ones.

2. **Redundancy with hybrid.**  If Phase 3 (hybrid refinement) runs for
   many iterations anyway, it may wash out the identity pre-training
   entirely — similar to how incremental init was neutral at 50 iterations.

3. **Overhead.**  Phase 1 adds computation.  If it doesn't produce
   substantially better encoders than random, it's pure waste.

## Implementation notes

- The identity training loop parallels `fit_hybrid` exactly, just with
  different targets and a temporary decoder that maps hidden activities
  (not output activities) back to x.
- Dimension handling: identity decoders have shape [n_neurons, d_in],
  actual decoders have shape [n_neurons, d_out].  These are separate
  tensors.
- Data-driven biases should still be used (centers=x for hidden layer 0).
- The identity phase could use early stopping: monitor reconstruction
  loss, stop when it plateaus.
- Expected implementation: ~50 lines added to `networks.py`, new
  `fit_identity_first()` method.

## Connection to existing work

This is analogous to **unsupervised pre-training** in deep learning
(Hinton et al. 2006, Bengio et al. 2007), where autoencoders or RBMs
learn good representations before supervised fine-tuning.  The NEF version
is more principled: the representation/transformation split is built into
the framework's theoretical foundations.

## Experimental plan (if revisited)

1. Implement `fit_identity_first()` with Variant B (full-network identity)
2. Sweep identity iterations: 10, 20, 50
3. Compare against baseline hybrid (50 iters) and hybrid→E2E
4. Try identity → hybrid refinement (Phase 1 + Phase 3) with 20+30 iters
5. Datasets: MNIST, Fashion-MNIST, CIFAR-10
