# leenef

Supervised learning with the Neural Engineering Framework (NEF) of
Eliasmith and Anderson,
using rate-based neurons on PyTorch.

## Overview

A standard neural network trains all weights with gradient descent.  NEF
takes a different approach: input weights (encoders) are random and fixed,
and output weights (decoders) are solved analytically via regularised
least-squares.  No gradient descent, no epochs, no learning rate — a single
layer trains in under 2 seconds.

Each neuron computes:

```
activity = |gain · ((x − d) · e)|
```

where **e** is a random unit vector (encoder direction), **d** is a
reference point sampled from training data (center), and **gain** is a
positive constant.  The neuron measures how the input deviates from a known
reference along a random direction — an unsigned distance, since the
absolute value responds to deviations in either direction.

Biases are derived from centers as `bias = −gain · (d · e)`, so there is no
separate bias distribution to tune.  Encoders are unit vectors on the
hypersphere; centers are sampled from training data.

The library provides `NEFLayer` for single-layer models, `NEFNetwork`
for multi-layer models, and `RecurrentNEFLayer` for temporal sequences,
all plugging into standard PyTorch workflows:

```python
from leenef.layers import NEFLayer
from leenef.networks import NEFNetwork
from leenef.recurrent import RecurrentNEFLayer

# Single layer — analytic solve, no gradient descent
layer = NEFLayer(d_in=784, n_neurons=2000, d_out=10, centers=x_train)
layer.fit(x_train, y_train)
predictions = layer(x_test)

# Multi-layer — six training strategies
net = NEFNetwork(d_in=784, d_out=10, hidden_neurons=[1000],
                 output_neurons=2000, centers=x_train)
net.fit_greedy(x, targets)
net.fit_hybrid(x, targets)
net.fit_target_prop(x, targets)
net.fit_end_to_end(x, targets)
net.fit_hybrid_e2e(x, targets)
net.fit_target_prop_e2e(x, targets)

# Recurrent / temporal model
rec = RecurrentNEFLayer(d_in=28, n_neurons=2000, d_out=10,
                        d_state=28, centers=x_train_seq)
rec.fit_hybrid_e2e(seq, targets, n_iters=10, n_epochs=20)
temporal_predictions = rec(x_test_seq)
```

In a multi-layer `NEFNetwork`, hidden layers encode only (their neuron
activities become the next layer's input) and only the output layer decodes.
Six training strategies are available.  **Greedy** solves each layer
independently with random encoders and analytic decoders — no gradient
computation at all.  **Hybrid** alternates analytic decoder solves with
gradient updates to encoder weights, learning useful encoder orientations
without full backprop.  **Target propagation** (`fit_target_prop`) replaces
backpropagation with layer-local targets: at each iteration it solves
representational decoders (the NEF inverse model) analytically at every
layer, then propagates targets backward via difference target propagation
and updates encoders with single-layer gradients only — no gradient flows
between layers.  **End-to-end** runs standard SGD on all parameters,
initialised from a greedy NEF solve, using NEF as an initialisation strategy
rather than a training method.  **Hybrid→E2E** (`fit_hybrid_e2e`) combines the
two: hybrid first learns good encoder orientations, then E2E refines all
parameters including decoders — the best overall strategy in the benchmark
table below.  **TP→E2E** (`fit_target_prop_e2e`) uses the local TP phase as the
warm start before a short global refinement pass.

`RecurrentNEFLayer` brings the same decode-then-re-encode idea to temporal
sequences.  It currently supports greedy, hybrid, target propagation through
time, end-to-end, and hybrid→E2E training.  Recurrent models default to
**relu** for stable gradient flow and expose an experimental
`state_target="predictive"` option for analytic greedy / TP state decoders.

## Setup

Requires Python 3.12+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

## Tests

```bash
pytest                                     # full suite
pytest -k test_fit_identity -q             # single test
```

## Benchmarks

> All timings are from a CPU-only setup (AMD Ryzen 5 PRO 5650U, no GPU).

Default configuration: **abs** activation, **hypersphere** encoders,
**per-neuron gain** U(0.5, 2.0), **data-driven biases** (`centers=x_train`),
Tikhonov solver (α = 0.01).  Recurrent layers default to **relu** because
abs has gradient ±1 everywhere, causing gradient explosion through BPTT.

```bash
python benchmarks/run.py --datasets mnist fashion_mnist cifar10 \
       --neurons 500 1000 2000 5000 --regression --seed 0 \
       --save-json results/feedforward.json
python benchmarks/run.py --datasets mnist fashion_mnist cifar10 \
       --neurons 2000 --multi --mlp --seed 0 \
       --save-csv results/feedforward-multi.csv
python benchmarks/run_recurrent.py --mode row \
       --strategies greedy hybrid target_prop e2e hybrid_e2e --seed 0 \
       --save-json results/recurrent-row.json
```

Both benchmark harnesses take `--seed` (use `0` for comparable reruns) and can
persist structured JSON / CSV results, which makes later plotting and README
table refreshes reproducible instead of purely manual.

### Single-layer results

#### Scaling with neuron count

| Dataset       |  500   | 1000   | 2000   | 5000   | 10k    | 20k    | 30k    |
|---------------|--------|--------|--------|--------|--------|--------|--------|
| MNIST         | 92.1%  | 94.3%  | 95.5%  | 96.9%  | 97.4%  | 97.9%  | 98.3%  |
| Fashion-MNIST | 82.6%  | 84.7%  | 85.7%  | 87.1%  | 88.4%  | 89.3%  | 89.8%  |
| CIFAR-10      | 43.7%  | 45.9%  | 47.8%  | 50.4%  | 51.0%  | 51.5%  | 51.8%  |
| Time          | <1s    | 1s     | 2s     | 10s    | 43s    | 140s   | 394s   |

At 2000 neurons, MNIST reaches 95.5% in ~2 seconds — within 3% of a
fully-trained MLP that takes 40× longer.  Performance scales monotonically
with neuron count but with severe diminishing returns: 30000 neurons at
~394 seconds reaches 98.3% on MNIST — close to hybrid's 98.5% at 315
seconds, but unable to match it despite using 10× more neurons.  The
single-layer ceiling on Fashion-MNIST (89.8%) and CIFAR-10 (51.8%) falls
further short of multi-layer results (91.0% / 58.1%), showing that learned
features are essential where brute-force neuron scaling cannot compensate.

#### Why data-driven biases matter (2000 neurons, abs activation)

|               | hyper  | + data | gauss  | + data | sparse | + data |
|---------------|--------|--------|--------|--------|--------|--------|
| MNIST         | 93.4%  |**95.6%**| 96.0% | 95.7%  | 95.6%  | 95.6%  |
| Fashion-MNIST | 84.1%  |**85.9%**| 86.0% | 86.0%  | 86.0%  | 85.6%  |
| CIFAR-10      | 45.9%  |**48.3%**| 47.3% | 47.5%  | 47.5%  | 48.2%  |

Without data-driven biases, hypersphere encoders lag Gaussian and sparse by
2–3%.  Data-driven biases **close the entire gap**.  The advantage of
Gaussian encoders was their varying norms creating an implicit distribution
of activation thresholds — data-driven biases make this explicit.  With
data biases, all encoder types converge to similar accuracy; the encoder
direction distribution no longer matters, only having enough random
directions does.

This is why the default uses hypersphere encoders (clean unit vectors,
principled random directions) plus data-driven biases (optimal threshold
placement) — rather than relying on Gaussian norms as an accidental proxy.

#### Activation comparison (2000 neurons, hypersphere, data biases)

| Activation | MNIST  | Fashion | CIFAR-10 |
|------------|--------|---------|----------|
| abs        |**95.7%**|**85.8%**|**48.1%**|
| relu       | 95.4%  | 85.3%   | 47.9%   |
| softplus   | 90.9%  | 82.4%   | 44.2%   |
| lif_rate   | 88.9%  | 81.2%   | 38.8%   |

Data-driven biases amplify the effect of activation choice.  With random
biases (not shown), all four activations cluster within ~1% of each other.
With data biases, neurons have more structured activation patterns with
sharper boundaries.  The abs and ReLU activations handle this well, but
softplus loses 5% on MNIST and lif_rate loses 7%.

The abs activation is a natural fit for the distance interpretation: each
neuron computes `|gain · ((x − d) · e)|`, responding to deviations in
either direction.  This doubles representational capacity compared to
ReLU, which discards one half of the encoding space.

#### Regression — California Housing (MSE, normalised targets)

| Neurons | Train MSE | Test MSE |
|---------|-----------|----------|
| 500     | 0.269     | 0.265    |
| 1000    | 0.254     | 0.249    |
| 2000    | 0.235     | 0.236    |
| 5000    | 0.213     | 0.227    |

More neurons improve accuracy with diminishing returns.  At 2000 neurons
the model generalises well, with test MSE within 1% of training MSE.

### Multi-layer results (hidden=[1000], output=2000)

| Model             | MNIST    | Fashion  | CIFAR-10 | Time (MNIST) |
|-------------------|----------|----------|----------|--------------|
| Linear baseline   | 85.3%    | 81.0%    | 39.6%    |     2s       |
| NEFLayer          | 95.7%    | 85.9%    | 47.8%    |     2s       |
| NEFNet-greedy     | 95.0%    | 85.4%    | 45.6%    |     3s       |
| NEFNet-hybrid     | 98.6%    | 90.2%    | 52.7%    |   318s       |
| NEFNet-target-prop| 98.6%    | 90.1%    | 51.0%    |   378s       |
| NEFNet-e2e        | 98.5%    | 90.3%    | 58.5%    |   241s       |
| NEFNet-hybrid→E2E |**98.6%** |**90.6%** | 58.4%    |   402s       |
| NEFNet-TP→E2E     | 98.5%    | 90.6%    |**58.5%** |   464s       |
| MLP (2×1000)      | 98.5%    | 89.7%    | 52.7%    |    83s       |

All multi-layer models use propagated data-driven biases: training data is
forwarded through each layer, and the resulting activations are used as
centers for the next layer's bias computation.  This is especially important
for greedy, which has no gradient learning — propagated centers lifted
greedy from 94.0% → 95.0% on MNIST and 84.0% → 85.4% on Fashion-MNIST.

The default hybrid configuration uses 50 iterations with α = 10⁻³ for the
decoder solver.  These were found via a systematic sweep over iterations
(10–100), solver regularisation (10⁻⁵–10⁻²), solver type (Tikhonov vs
Cholesky vs unregularised lstsq), hidden layer count, and neuron counts.

**Iterations dominate all other hyperparameters.** Going from 10 to 50
iterations lifts hybrid from 97.2% → 98.5% on MNIST, 87.9% → 90.4% on
Fashion, and 45.9% → 53.3% on CIFAR-10.  More layers, more neurons per
layer, and switching solvers each contribute less than 0.2%.  Returns
diminish past 50 iterations (75 and 100 barely improve).

**Lower decoder regularisation helps hybrid** but is dataset-dependent.
α = 10⁻³ is the sweet spot across all three datasets.  Going lower
(10⁻⁴) hurts Fashion and CIFAR-10 — the decoder overfits the current
encoder state, producing noisy gradients that destabilise encoder learning.
At α = 10⁻⁵ results collapse entirely (96.4% MNIST, 32.7% CIFAR-10).

**Warm-started E2E remains a strong option.**  `fit_target_prop_e2e`
reaches 98.5% / 90.6% / 58.5% with a gentler `eta=0.01` TP warm start,
keeping it competitive with `fit_hybrid_e2e`, which lands at
98.6% / 90.6% / 58.4%.  The hybrid→E2E warm start keeps the slight
MNIST/Fashion edge, while TP→E2E still edges ahead on CIFAR-10.

**Plain target propagation benefits from a moderate fixed step.**
`fit_target_prop` still avoids cross-layer backprop entirely: it solves
representational decoders analytically, propagates DTP targets backward, and
updates each layer with only single-layer gradients.  The current default keeps
those activity targets unconstrained and uses `eta=0.03`, which gave the best
overall plain-TP trade-off in the seeded reruns: 98.6% / 90.1% / 51.0%.
Strict target projection (`project_targets=True`) is still available for
experiments, but a later seeded full plain-TP rerun with projection fell to
98.3% / 88.4% / 48.9%, so projection remains opt-in.  Adaptive
infeasible-target backoff (`max_infeasible_fraction=...`) also remains
available for experiments, and the new `eta_schedule` /
`hidden_max_infeasible_fraction` controls widen that experimental surface.
However, representative `3k` / `1k` seeded slice searches still did not produce
a clean across-seed replacement for the fixed plain-TP default: the best hidden
feasibility budget (`0.01`) was essentially tied with the baseline on average,
so the fixed defaults remain the published path.

**Target propagation eta sweep** (historical broad sweep; 50 iterations,
lr=10⁻³, normalised step):

> Measured with an earlier configuration (gain=1.0, data-driven biases).
> We no longer use it as a literal tuning table: the current seeded reruns
> split the defaults at `eta=0.03` for plain TP and `eta=0.01` for the TP→E2E
> warm start.  The table is still useful for the qualitative pattern:
> MNIST is forgiving, Fashion prefers a mid-sized step, and CIFAR-10 wants a
> gentler one.

| eta    | MNIST | Fashion | CIFAR-10 |
|--------|-------|---------|----------|
| 0.001  | —     | —       | 53.8%    |
| 0.002  | —     | 89.0%   | **54.8%**|
| 0.005  | —     | 89.6%   | 54.7%   |
| 0.01   | 98.5% | 90.0%   | 52.6%    |
| 0.05   | 98.5% | **90.2%**| 51.1%   |
| 0.10   | **98.6%** | 90.0% | 51.0%  |

MNIST is insensitive to eta across a wide range.  Fashion prefers a mid-sized
step, while CIFAR-10 benefits from gentler updates.  That is why the current
defaults no longer share one TP step across both strategies: plain TP uses
`eta=0.03`, while TP→E2E keeps a smaller `eta=0.01` warm start.  The optional
projection/adaptive controls remain useful for experiments, but the simple
fixed defaults above were more reliable on the current seeded reruns.

See `docs/analytical_target_propagation.md` for the full algorithm and
analysis of related approaches.

Greedy multi-layer is still slightly worse than single-layer (95.0% vs
95.5% on MNIST), since a random nonlinear re-encoding loses some
information.  But propagated data-driven biases dramatically narrow the
gap — from 1.5% to 0.5% on MNIST — by ensuring every layer's neurons are
centred around realistic activation patterns rather than random points.

#### Hybrid improvement sweep

> Measured with an earlier configuration (gain=1.0, data-driven biases).
> Qualitative conclusions (CE is catastrophic, flat LR is optimal) are
> expected to hold with current defaults.

We also tested cross-entropy loss for encoder gradients, cosine LR
scheduling, incremental hidden-layer initialisation (warm-start from a
solved single-layer), and mini-batch gradient steps.  None improved on the
MSE full-batch baseline:

| Variant              | MNIST  | Fashion | CIFAR-10 |
|----------------------|--------|---------|----------|
| Baseline (MSE, flat) | 98.65% | 90.15%  | 52.67%   |
| CE loss              | 94.18% | 82.02%  | 37.23%   |
| Cosine schedule      | 98.34% | 89.98%  | 50.95%   |
| Incremental init     | 98.60% | 90.10%  | 52.86%   |
| Mini-batch (256, 3)  | 96.05% | 85.82%  | 38.71%   |

CE loss is catastrophic: the analytic decoder solve targets MSE (outputs
near 0/1 for one-hot targets), but cross-entropy interprets these as
logits, creating a destructive conflict.  Mini-batch hurts because
3 mini-batch steps provide far less gradient coverage than one full-batch
step.  Cosine annealing decays too aggressively — hybrid's decoder
re-solve already stabilises each iteration, making a flat LR optimal.
Incremental init is neutral: 50 iterations absorb the warm-start advantage.

#### Activation effect on multi-layer hybrid

| Activation | MNIST  | Fashion |
|------------|--------|---------|
| relu       |**97.8%**|**88.7%**|
| abs        | 97.5%  | 88.0%   |
| lif_rate   | 95.2%  | 85.1%   |
| softplus   | 94.0%  | 84.1%   |

> Measured with 10 hybrid iterations, gain=U(0.5, 2.0), data-driven
> biases, α = 10⁻².

With data-driven biases and per-neuron gain, relu slightly edges out abs
for multi-layer hybrid training.  Per-neuron gain diversity means ReLU's
zero-gradient region no longer kills half the neurons uniformly — some
neurons have low gain (wide tuning) while others have high gain (narrow
tuning), creating a rich gradient landscape.  The ranking reversal from
single-layer (where abs leads) suggests that gradient flow through
multiple encode–decode cycles benefits from ReLU's sparsity.

## Temporal / recurrent models

`RecurrentNEFLayer` extends the feedforward pipeline with the canonical NEF
decode-then-re-encode feedback loop.  At each timestep *t*:

```
x_aug = concat(u[t], s[t-1])                   # (B, d_in + d_state)
a[t]  = activate(gain · (x_aug · E^T) + bias)  # (B, n_neurons)
s[t]  = a[t] @ D_state                         # (B, d_state)   ← state decoder
y     = a[T] @ D_out                           # (B, d_out)     ← output decoder
```

The **state decoder** `D_state` is the NEF "representational decoder" — it
extracts a low-dimensional state summary from the population at each step and
feeds it back through the encoders.  The **output decoder** `D_out` is the
"transformational decoder" applied at the final timestep to produce the task
prediction.  The dynamics matrix is implicit in the encoder weights that
project both external input and feedback state into neuron space.

**Recurrent layers default to relu activation** rather than abs.  The abs
activation has gradient ±1 everywhere (no sparsity), which causes gradient
explosion through BPTT over many timesteps.  ReLU's zero gradient on negative
activations provides the damping needed for stable recurrent gradient flow.

```python
from leenef.recurrent import RecurrentNEFLayer

# Default: relu activation (abs explodes through recurrent BPTT)
layer = RecurrentNEFLayer(d_in=28, n_neurons=2000, d_out=10,
                          d_state=28, centers=x_train_seq)

# Training strategies
layer.fit_greedy(seq, targets, n_iters=5)
layer.fit_hybrid(seq, targets, n_iters=10, lr=1e-3)
layer.fit_target_prop(seq, targets, n_iters=50, lr=1e-3, eta=0.1,
                      state_target="predictive")  # experimental recurrent TP target
layer.fit_end_to_end(seq, targets, n_epochs=50, lr=1e-3, grad_clip=1.0)
layer.fit_hybrid_e2e(seq, targets, n_iters=10, n_epochs=20, e2e_lr=1e-3)

# Inference: (B, T, d_in) → (B, d_out)
predictions = layer(x_test_seq)
```

Recurrent hybrid and E2E-style training support gradient clipping via
`grad_clip` (default `1.0`) because long unrolled sequences are otherwise more
prone to gradient explosion than the feed-forward models.

### Recurrent results — Sequential MNIST

Row-by-row sMNIST: each image is a sequence of 28 rows (T=28, d=28),
classified at the final timestep.  All NEF models use 2000 neurons,
relu activation, gain U(0.5, 2.0), and data-driven biases.

Best available full-row references:

| Model                           | Test accuracy | Time    | Source |
|---------------------------------|---------------|---------|--------|
| RecNEF-greedy (5 iter)          |  15.3%        |   241s  | historical full run |
| RecNEF-hybrid (10 iter)         |  21.0%        |   461s  | seeded full rerun |
| RecNEF-target-prop (50 iter)    |  12.1%        |  5864s  | historical full run |
| RecNEF-E2E (50 epochs)          |  98.5%        |   799s  | seeded full rerun |
| RecNEF-hybrid→E2E (10 iter + 20 epochs)| **98.6%**|   686s  | seeded full rerun |
| LSTM-128 (20 epochs)            |  98.3%        |    98s  | historical full run |

The seeded reruns make the recurrent picture much clearer than the old
spot-checks: `fit_hybrid_e2e` is now the strongest recurrent result we have,
edging pure recurrent E2E while training faster on the full row-wise rerun.
Plain recurrent hybrid still collapses, so the reliable recurrent choices are
currently `fit_end_to_end` and `fit_hybrid_e2e`.

### Predictive decoded-state targets (experimental recurrent TP)

The current recurrent TP implementation now exposes three experimental controls:
`state_target` for the analytic state decoder, `auxiliary_weight` for optional
pre-final label supervision, and `project_targets` for activity-space target
projection.  The default `reconstruction` state target asks the state decoder
to reproduce the current input frame.  The experimental `predictive` target
instead decodes the next frame (and zeros at the terminal step), which is a
closer fit to what the recurrent state should carry.

Seeded row-wise sMNIST TP slice (`2k` train / `1k` test, `10` TP iterations,
`2000` neurons):

| State target    | Seed 0 | Seed 1 | Seed 2 | Mean |
|-----------------|--------|--------|--------|------|
| Reconstruction  | 22.2%  | 22.6%  | 21.8%  | 22.2% |
| Predictive      | 31.9%  | 30.9%  | 32.8%  | 31.9% |

This is still far below recurrent E2E / hybrid→E2E, but it is the first
recurrent TP change that improves accuracy consistently across seeds, so it is
worth keeping as an active research direction rather than a dead-end branch.
The same direction held on a larger seeded `5k` / `1k` slice, where predictive
targets improved recurrent TP from 23.0% to 32.8%.

Follow-up on that predictive path showed a more nuanced picture for the other
experimental controls:

- explicit auxiliary label supervision on earlier timesteps did **not** help;
  a seed-0 sweep over `auxiliary_weight ∈ {0.5, 1, 2, 4}` all underperformed
  the no-auxiliary baseline
- activity-target projection, which hurt the feed-forward TP default path,
  looks mildly positive here when combined with predictive state targets

| Predictive TP variant | Seed 0 | Seed 1 | Seed 2 | Mean |
|-----------------------|--------|--------|--------|------|
| No projection         | 31.9%  | 30.9%  | 32.8%  | 31.9% |
| Projected targets     | 32.5%  | 30.6%  | 33.1%  | 32.1% |

On the larger seeded `5k` / `1k` slice, projected predictive TP again improved
slightly from 32.8% to 33.3%.  That makes predictive state targets the main
recurrent TP gain, projected targets a smaller follow-on improvement, and
auxiliary timestep labels an experiment worth keeping opt-in only.

### Why abs fails for recurrence

In feedforward, abs doubles
representational capacity by responding to both directions.  In recurrent
BPTT, abs has gradient sign(x) ∈ {−1, +1} at every neuron, meaning the
recurrent Jacobian has no sparsity to damp gradient magnitudes.  Over 28
timesteps, this causes gradient explosion regardless of gain or learning
rate.  ReLU's zero gradient on negative activations (roughly half the
neurons) provides the critical damping for stable recurrent learning.
Tested: E2E with abs gets 10.1% (random), E2E with relu gets 98.5%.

## Visualisations

Generate plots with `python benchmarks/plot.py` (requires the `bench` extras:
`matplotlib` and `numpy`).  The checked-in plots are still summary views built
from the repo's tracked benchmark tables, but the benchmark harnesses can now
also save JSON / CSV outputs for reproducible reruns and future plot refreshes.

![Neuron scaling](docs/neuron_scaling.png)
![Data-driven bias effect](docs/bias_effect.png)
![Strategy comparison and speed–accuracy trade-off](docs/strategy_comparison.png)
![Activation effect on multi-layer hybrid](docs/activation_multilayer.png)

## Conclusions

1. **Data-driven biases are the key design choice.**  Rewriting the encoding
   as `|gain · ((x − d) · e)|` reveals each neuron measures unsigned
   deviation from a reference point *d* along direction *e*.  Sampling *d*
   from training data closes the entire 2–3% gap between encoder types and
   makes the encoder direction distribution irrelevant — only having enough
   random directions matters.

2. **Single-layer NEF is remarkably effective but hits a ceiling.**  With
   2000 neurons and a 2-second analytic solve, it reaches 95.7% on MNIST
   and 85.9% on Fashion-MNIST — within 3% of a fully-trained MLP on
   MNIST.  Even
   with 30000 neurons and 394 seconds of compute (comparable to hybrid),
   single-layer tops out at 98.3% / 89.8% / 51.8% — learned features in
   multi-layer networks cannot be replaced by brute-force neuron scaling.

3. **The abs activation is a natural fit for single-layer models.**
   Computing an unsigned distance along the encoder direction doubles
   representational capacity by responding to deviations in either
   direction.  With data-driven biases, abs is the best single-layer
   activation.  For multi-layer hybrid training, relu slightly edges
   ahead — per-neuron gain diversity means ReLU's sparsity aids gradient
   flow without killing neurons uniformly.

4. **Activation sensitivity is controlled by bias structure.**  With random
   biases, all activations perform within ~1%.  Data-driven biases create
   sharper activation patterns that reward sharp-threshold activations
   (abs, ReLU) and punish smooth ones (softplus −5%, lif_rate −7% on
   MNIST).

5. **Hybrid→E2E remains the best balanced feed-forward strategy.**  Running
   hybrid then E2E reaches 98.6% / 90.6% / 58.4% — best on Fashion-MNIST,
   tied for best on MNIST within rounding, and only 0.1 points behind
   TP→E2E on CIFAR-10.  The hybrid phase learns encoder orientations with
   analytic decoders; the E2E phase unlocks full gradient training without
   giving back the strong Fashion result.

6. **Hybrid alone remains a strong analytic-gradient compromise.**  With 50
   iterations and α = 10⁻³, pure hybrid reaches 98.6% / 90.2% / 52.7%
   while preserving analytic decoders.  Iteration count dominates the
   other hybrid hyperparameters.

7. **Decoder regularisation is the second lever for hybrid.**  α = 10⁻³
   is optimal — lower values let decoders overfit the current encoder
   state, producing noisy gradients; higher values underfit.  At α = 10⁻⁵
   the training collapses entirely.

8. **CE loss is incompatible with hybrid's analytic decoders.**  Decoders
   solve for MSE-optimal outputs near 0/1; cross-entropy interprets these
   as logits, creating a destructive gradient conflict that drops CIFAR-10
   from 53% to 37%.  The loss used for encoder gradients must match the
   decoder objective.

9. **Propagated data-driven biases rescue greedy.**  Without them, greedy
   multi-layer was worse than single-layer (94.0% vs 95.5% MNIST).
   Forwarding training data through each layer and using activations as
   centers for the next layer's biases narrows the current seeded gap to
   0.7%, though
   greedy still cannot match gradient-trained strategies.

10. **Per-neuron gain U(0.5, 2.0) is the default.**  In canonical NEF,
    each neuron has its own gain sampled from a distribution.  Per-neuron
    gain via `gain=(lo, hi)` is now the default for all layers and
    strategies, following the NEF tradition of diverse tuning curves.
    The impact is modest in feedforward networks (gradient-based methods
    adapt encoders to compensate), but it provides a better initial
    encoding — each neuron responds at a different sensitivity level,
    giving the population a richer representation from the start.

11. **Target propagation stays competitive without cross-layer backprop.**
   Using NEF representational decoders as analytical inverse models and a
   normalised activity-space step, plain TP reaches 98.6% / 90.1% / 51.0%
   using only single-layer gradients.  A short TP→E2E fine-tune pushes
   that to 98.5% / 90.6% / 58.5%.  The current seeded defaults now split
   `eta` by strategy — `0.03` for plain TP, `0.01` for TP→E2E — because a
   single shared step was not the best trade-off.  Strict target
   projection plus the newer eta-schedule / hidden-feasibility controls
   remain opt-in experimental paths rather than the default one.

12. **Recurrent warm starts are now real full-scale results, not just
    spot-check curiosities.**  On seeded full row-wise sMNIST reruns, recurrent
    hybrid→E2E reaches 98.6% and pure recurrent E2E reaches 98.5%, while
    plain recurrent hybrid remains near chance.  That makes the warm-start
    path the best current recurrent story.

13. **Predictive decoded-state targets are the first recurrent TP change
    that clearly helps.**  On seeded row-wise sMNIST TP slices, switching
    the analytic state decoder from reconstruction targets to predictive
    targets lifts mean accuracy from 22.2% to 31.9%, and the same effect
    persisted on a larger `5k` / `1k` slice (23.0% → 32.8%).  Projected
    activity targets add a smaller follow-on lift on top of that predictive
    path (31.9% → 32.1% mean on the `2k` / `1k` seeds, 32.8% → 33.3% on the
    larger slice), while auxiliary timestep label supervision did not improve
    the predictive baseline.  Recurrent TP is still research-grade, but this
    is a meaningful step forward.

## Divergences from canonical NEF

This implementation adapts the NEF framework for supervised learning and
departs from the original formulation in several ways:

- **Scalar gain per neuron, not gain–intercept pairs.**  Canonical NEF
  samples (gain, intercept) pairs to produce diverse tuning curves.  We
  derive intercepts from data-driven centers instead, and support per-neuron
  gain distributions via `gain=(lo, hi)` though uniform gain works well.

- **Absolute value activation, not spiking neurons.**  We use rate-based
  activations (default: `abs`) rather than spiking neuron models.  The abs
  activation gives each neuron a two-sided tuning curve — it responds to
  deviations in either direction from its preferred point.

- **Encode-only hidden layers.**  In canonical NEF, every population has
  decoders that extract a representation or transformation.  Our hidden
  layers pass raw activities to the next layer; only the output layer
  decodes.  This treats hidden layers as learned representations rather
  than explicit transformations.

- **Random fallback biases.**  When no training data centers are provided,
  biases fall back to iid Gaussian samples rather than a principled
  intercept distribution.  Always pass `centers=x_train` for best results.

## Related work

A single NEF layer — random input weights, nonlinear activation, analytic
output weights — is architecturally identical to an Extreme Learning Machine
(ELM; Huang et al., 2006).  Both are instances of the random features
framework (Rahimi & Recht, 2007), which shows that random projections
followed by a nonlinearity approximate kernel functions, with the number of
random features controlling approximation quality.  NEF arrives at the same
architecture from neuroscience principles: encoders model neuron preferred
directions, the activation models firing rates, and decoders recover the
represented signal.  The data-driven bias interpretation developed here —
each neuron measures deviation from a training sample — adds a connection
to radial basis function networks, where each basis function is centred on
a data point.

## Components

| Module | Purpose |
|--------|---------|
| `leenef/encoders.py` | Random encoder generation (hypersphere, Gaussian, sparse) |
| `leenef/activations.py` | Rate-based neuron models (ReLU, softplus, LIF rate, abs) |
| `leenef/solvers.py` | Decoder solvers (lstsq, Tikhonov, Cholesky, normal equations) |
| `leenef/layers.py` | `NEFLayer(nn.Module)` — encode → activate → decode |
| `leenef/networks.py` | `NEFNetwork(nn.Module)` — multi-layer with greedy / hybrid / target-prop / e2e / hybrid→E2E / TP→E2E |
| `leenef/recurrent.py` | `RecurrentNEFLayer(nn.Module)` — temporal decode-then-re-encode loop with greedy / hybrid / target-prop / e2e / hybrid→E2E |
| `benchmarks/run.py` | Benchmark harness with single-layer, multi-layer, and MLP baselines |
| `benchmarks/run_recurrent.py` | Sequential MNIST benchmark for recurrent NEF and LSTM baseline |
| `benchmarks/plot.py` | Visualisation script (generates `docs/*.png`) |
| `docs/analytical_target_propagation.md` | Algorithm derivation, comparison, and survey of related approaches |

## References

- C. Eliasmith & C. H. Anderson, *Neural Engineering: Computation,
  Representation, and Dynamics in Neurobiological Systems*, MIT Press, 2003.
  [MIT Press](https://mitpress.mit.edu/9780262550604/)
- C. Eliasmith, "A unified approach to building and controlling spiking
  attractor networks", *Neural Computation* 17(6), 2005.
  [doi:10.1162/0899766053429390](https://doi.org/10.1162/0899766053429390)
- G.-B. Huang, Q.-Y. Zhu & C.-K. Siew, "Extreme learning machine: Theory
  and applications", *Neurocomputing* 70(1–3), 2006.
  [doi:10.1016/j.neucom.2005.12.126](https://doi.org/10.1016/j.neucom.2005.12.126)
  — the closely related random-feature approach from the ML side.
- A. Rahimi & B. Recht, "Random Features for Large-Scale Kernel Machines",
  *NeurIPS*, 2007.
  [paper](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)

**Datasets:**
- [MNIST](http://yann.lecun.com/exdb/mnist/) — Y. LeCun et al., 1998.
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — H. Xiao et al., 2017.
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — A. Krizhevsky, 2009.
- [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) — R. K. Pace & R. Barry, 1997 (via scikit-learn).
