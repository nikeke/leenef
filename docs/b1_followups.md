# B1 follow-up experiment specs (queued for after B2)

These are the two H1-completion experiments the plan promised in §6 R5.x
and Plan §5/B1.  Both verify the **DC-component / dataset-centeredness**
mechanism that explains why `data_driven == zero` on MNIST/Fashion but
`data_driven > zero` by ~1.4 pp on CIFAR-10.

**Predictions:**

- B1.4 (per-channel-centred CIFAR-10): the `data_driven – zero` gap
  *closes* to within 0.2 pp.  This is the falsification test for the
  DC mechanism.
- B1.5 (de-centred MNIST): the `data_driven – zero` gap *appears* on
  the order of +1 to +1.5 pp.  The mirror falsification test.

Together, B1.4 + B1.5 → the H1 mechanism is no longer a hypothesis;
it's a measured-and-confirmed claim.

## B1.4 — `biases_cifar10_centered`

Identical to `biases_cifar10` except the dataset preprocessing
subtracts the **per-channel** training-set mean (3-vector) before
flattening to 3072 dims.

Implementation sketch:

1. In `benchmarks/sl_clean/datasets.py` (or wherever CIFAR-10 is
   loaded), add a `center_per_channel: bool = False` flag.  When
   true, compute `mean[c] = x_train[:, c, :, :].mean()` over the
   training split, then do `x = x - mean[None, :, None, None]`.
   Hash the centred tensor to populate `processed_hashes`.
2. New experiment in `experiments.py`:

   ```python
   def _biases_cifar10_centered_context() -> dict:
       return _make_biases_context(
           dataset="CIFAR10",
           n_train=50000,
           n_test=10000,
           d_in=3072,
           preprocess={"center_per_channel": True},
       )()

   register(ExperimentSpec(
       name="biases_cifar10_centered",
       run_fn=_biases_run,
       default_run_context=_biases_cifar10_centered_context,
       primary_metric="test_accuracy",
       higher_is_better=True,
       description=_BIASES_DESCRIPTION + " (CIFAR-10, per-channel-centred)",
   ))
   ```
3. `_make_biases_context` may need a small `preprocess` plumbing
   change so the dataset loader applies it.  Result JSON keeps the
   `dataset_metadata.processed_hashes` distinct from the un-centred
   variant so downstream analysis can tell them apart.
4. Same 10-seed budget, same n=5000, same 9 cells.  ~25 min CPU.

## B1.5 — `biases_mnist_decentered`

Identical to `biases_mnist` except the dataset is *added to* by a
strong DC offset (e.g. +0.5 in [0,1] pixel space, clamped to [0,1])
*before* flattening.  This synthesizes a CIFAR-10-like DC profile on
top of MNIST.

Concrete preprocessing:

```
x_train_decentered = (x_train + 0.5).clamp(0.0, 1.0)
```

(The clamp keeps pixel values legal but slightly squashes the high
end; we accept this small distortion in exchange for a cleanly DC-
shifted dataset.  Alternative: `x_train + 0.5` un-clamped, with
input-range adjusted; pick whichever yields a cleaner signal in a
quick pilot.)

Hash the de-centred tensor distinctly.  10 seeds × 9 cells × n=5000.
~17 min CPU.

## Analyzer

Same `analyze_biases_panel.py` works on both.  Add a small driver
that emits the gap-comparison summary across MNIST / decentered-MNIST
/ CIFAR-10 / centred-CIFAR-10 in one table for the paper.

## When to run

After B2 panels finish and we commit the B2 deliverables.  These two
follow-ups are independent of the streaming work and can run
sequentially in ~45 min total CPU.

## What we'll write in §5.1

If predictions hold:

> "On centred datasets (MNIST, Fashion-MNIST), data-driven and zero
> biases agree to within 0.2 pp because the dataset's mean is already
> at the origin where zero biases place the abs fold.  On non-centred
> datasets (CIFAR-10), data-driven biases recover ~1.4 pp.  Two
> targeted controls confirm the mechanism: per-channel centring
> CIFAR-10 *removes* the gap, and adding a DC offset to MNIST
> *creates* one."

If a prediction *fails*: that's load-bearing too — the DC mechanism
isn't the whole story and §5.1 has to acknowledge it.
