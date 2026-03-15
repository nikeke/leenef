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

# Single layer — analytic solve, no gradient descent
layer = NEFLayer(d_in=784, n_neurons=2000, d_out=10, centers=x_train)
layer.fit(x_train, y_train)
predictions = layer(x_test)

# Multi-layer — three training strategies
net = NEFNetwork(d_in=784, d_out=10, hidden_neurons=[1000],
                 output_neurons=2000, centers=x_train)
net.fit_greedy(x, targets)
net.fit_hybrid(x, targets)
net.fit_end_to_end(x, targets)
```

In a multi-layer `NEFNetwork`, hidden layers encode only (their neuron
activities become the next layer's input) and only the output layer decodes.
Five training strategies are available.  **Greedy** solves each layer
independently with random encoders and analytic decoders — no gradient
computation at all.  **Hybrid** alternates analytic decoder solves with
gradient updates to encoder weights, learning useful encoder orientations
without full backprop.  **Target propagation** (`fit_target_prop`) replaces
backpropagation with layer-local targets: at each iteration it solves
representational decoders (the NEF inverse model) analytically at every
layer, then propagates targets backward via difference target propagation
and updates encoders with single-layer gradients only — no gradient flows
between layers.  **End-to-end** runs standard SGD on all parameters,
initialised from a greedy NEF solve, using NEF as an initialisation
strategy rather than a training method.  **Hybrid→E2E** (`fit_hybrid_e2e`)
combines the two: hybrid first learns good encoder orientations, then E2E
refines all parameters including decoders — the best overall strategy.

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
       --neurons 500 1000 2000 5000 --regression
python benchmarks/run.py --datasets mnist fashion_mnist cifar10 \
       --neurons 2000 --multi --mlp
```

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
| NEFLayer          | 95.6%    | 85.5%    | 47.8%    |     2s       |
| NEFNet-greedy     | 95.1%    | 85.5%    | 45.8%    |     3s       |
| NEFNet-hybrid     | 98.5%    | 90.0%    | 51.7%    |   315s       |
| NEFNet-target-prop| 98.6%    | 90.0%    | 53.9%    |   351s       |
| NEFNet-hybrid→E2E |**98.6%** |**91.0%** |**58.1%** |   412s       |
| NEFNet-e2e        | 98.4%    | 90.3%    | 57.8%    |   240s       |
| MLP (2×1000)      | 98.1%    | 90.2%    | 54.6%    |    84s       |

All multi-layer models use propagated data-driven biases: training data is
forwarded through each layer, and the resulting activations are used as
centers for the next layer's bias computation.  This is especially important
for greedy, which has no gradient learning — propagated centers lifted
greedy from 94.0% → 95.1% on MNIST and 84.0% → 85.5% on Fashion-MNIST.

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

**Hybrid→E2E is the best overall strategy.**  Running 50 hybrid iterations
then 20 E2E epochs (`fit_hybrid_e2e`) reaches 98.6% / 91.0% / 58.1% —
the highest accuracy on all three datasets.  The hybrid phase learns
good encoder orientations with analytic decoders; the E2E phase then
unlocks decoder learning to squeeze out the last gains.

**Target propagation avoids backprop entirely.**  `fit_target_prop` replaces
cross-layer gradient flow with analytical inverse models: each iteration
solves representational decoders (mapping activities back to inputs), then
uses difference target propagation (DTP) to compute local targets.  Encoder
updates use single-layer gradients only — no gradient ever flows between
layers.  A normalised gradient step ensures *eta* directly controls what
fraction of activity norm the targets deviate by, making the hyperparameter
scale-independent.  With the default eta=0.1, **TP matches or exceeds hybrid
on all three datasets** (98.6% / 90.0% / 53.9% vs 98.5% / 90.0% / 51.7%).
Per-dataset eta tuning can squeeze more from CIFAR-10 (see sweep below).
Key speed optimisations: skipping the
unused first-layer representational decoder, reusing the forward-pass
computation graph for local gradient steps, and caching the output-layer
Cholesky factorisation to solve both task and representational decoders
from one A^T A decomposition.

**Target propagation eta sweep** (50 iterations, lr=10⁻³, normalised step):

> Measured with an earlier configuration (gain=1.0, data-driven biases).
> Qualitative conclusions (CIFAR-10 prefers small eta, MNIST is insensitive)
> are expected to hold with current defaults.

| eta    | MNIST | Fashion | CIFAR-10 |
|--------|-------|---------|----------|
| 0.001  | —     | —       | 53.8%    |
| 0.002  | —     | 89.0%   | **54.8%**|
| 0.005  | —     | 89.6%   | 54.7%   |
| 0.01   | 98.5% | 90.0%   | 52.6%    |
| 0.05   | 98.5% | **90.2%**| 51.1%   |
| 0.10   | **98.6%** | 90.0% | 51.0%  |

MNIST is insensitive to eta across two orders of magnitude.  Fashion
slightly prefers larger eta (~0.05).  CIFAR-10 benefits dramatically from
smaller eta — 0.002 lifts test accuracy from 51.0% to 54.8%, surpassing
both hybrid (52.3%) and the MLP baseline (53.4%).  The pattern is clear:
harder data needs gentler target updates to keep targets in the feasible
activity region.

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

## Visualisations

Generate plots with `python benchmarks/plot.py` (requires matplotlib).

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
   2000 neurons and a 2-second analytic solve, it reaches 95.5% on MNIST
   and 85.7% on Fashion-MNIST — within 3% of a fully-trained MLP.  Even
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

5. **Hybrid→E2E is the best overall strategy.**  Running hybrid then E2E
   reaches 98.6% / 91.0% / 58.1% — the highest accuracy on all three
   datasets.  The hybrid phase learns encoder orientations with analytic
   decoders; the E2E phase unlocks full gradient training to close the
   CIFAR-10 gap.

6. **Hybrid alone surpasses both E2E and MLP on easy datasets.**  With 50
   iterations and α = 10⁻³, pure hybrid reaches 98.5% MNIST / 90.0%
   Fashion while preserving analytic decoders.  Iterations dominate all
   other hyperparameters.

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
   centers for the next layer's biases narrows the gap to 0.5%, though
   greedy still cannot match gradient-trained strategies.

10. **Per-neuron gain U(0.5, 2.0) is the default.**  In canonical NEF,
    each neuron has its own gain sampled from a distribution.  Per-neuron
    gain via `gain=(lo, hi)` is now the default for all layers and
    strategies, following the NEF tradition of diverse tuning curves.
    The impact is modest in feedforward networks (gradient-based methods
    adapt encoders to compensate), but it provides a better initial
    encoding — each neuron responds at a different sensitivity level,
    giving the population a richer representation from the start.

11. **Target propagation matches hybrid without backprop.**  Using NEF
    representational decoders as analytical inverse models and a normalised
    gradient step, TP matches or exceeds hybrid on all three datasets
    (98.6% / 90.0% / 53.9% vs 98.5% / 90.0% / 51.7%) using only
    single-layer gradients.  Per-dataset eta tuning (see sweep) can lift
    CIFAR-10 further.  The key insight: normalising the step so eta
    controls the fractional deviation keeps targets in the feasible
    activity region regardless of scale.

## Recurrent / temporal extension

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

# Training strategies (same four as feedforward, plus greedy)
layer.fit_greedy(seq, targets, n_iters=5)
layer.fit_hybrid(seq, targets, n_iters=10, lr=1e-3)
layer.fit_target_prop(seq, targets, n_iters=50, lr=1e-3, eta=0.1)
layer.fit_end_to_end(seq, targets, n_epochs=50, lr=1e-3)

# Inference: (B, T, d_in) → (B, d_out)
predictions = layer(x_test_seq)
```

### Recurrent results — Sequential MNIST

Row-by-row sMNIST: each image is a sequence of 28 rows (T=28, d=28),
classified at the final timestep.  All NEF models use 2000 neurons,
relu activation, gain U(0.5, 2.0), and data-driven biases.

| Model                           | Test accuracy | Time    |
|---------------------------------|---------------|---------|
| RecNEF-greedy (5 iter)          |  15.3%        |   241s  |
| RecNEF-hybrid (10 iter)         |   9.6%        |   454s  |
| RecNEF-target-prop (50 iter)    |  12.1%        |  5864s  |
| RecNEF-E2E (50 epochs)          | **98.5%**     |   802s  |
| LSTM-128 (20 epochs)            | **98.3%**     |    98s  |

**The feedforward NEF advantage does not transfer to recurrence.**  In
feedforward, random encoders work because each input is encoded
independently.  In the recurrent case, the state decoder feeds noisy
state estimates back into the encoders, compounding errors across
timesteps.  Greedy, hybrid, and target propagation all fail to learn
useful temporal dynamics — their per-iteration encoder updates cannot
overcome the cascading noise in the 28-step feedback loop.

**End-to-end BPTT is the only competitive strategy** (98.5%), matching
LSTM (98.3%).  Full gradient flow through all timesteps allows SGD to
jointly optimise encoders, biases, and both decoders, learning state
representations that remain coherent across the feedback loop.  The
greedy initialisation provides a starting point, but essentially all
learning happens during SGD.

**Why abs fails for recurrence:**  In feedforward, abs doubles
representational capacity by responding to both directions.  In recurrent
BPTT, abs has gradient sign(x) ∈ {−1, +1} at every neuron, meaning the
recurrent Jacobian has no sparsity to damp gradient magnitudes.  Over 28
timesteps, this causes gradient explosion regardless of gain or learning
rate.  ReLU's zero gradient on negative activations (roughly half the
neurons) provides the critical damping for stable recurrent learning.
Tested: E2E with abs gets 10.1% (random), E2E with relu gets 98.5%.

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
| `leenef/networks.py` | `NEFNetwork(nn.Module)` — multi-layer with greedy/hybrid/target-prop/e2e |
| `leenef/recurrent.py` | `RecurrentNEFLayer(nn.Module)` — temporal decode-then-re-encode loop |
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
