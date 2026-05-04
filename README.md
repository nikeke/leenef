# leenef

Machine learning with a rate-based specialization of the Neural
Engineering Framework (NEF) in PyTorch.

> **Repository status — SSUP trilogy rebuild.** The stable core lives in
> `src/leenef/` and `tests/`.  Active supervised-learning experiments and
> preserved artifacts live under `benchmarks/sl_clean/`,
> `results/sl_clean/`, and `docs/SSUP-I.md`; older pre-restart
> infrastructure remains in `attic/` for reference.

## Overview

leenef keeps the NEF encoder/decoder split but uses rate neurons and an
analytic decoder solve for machine learning:

```text
a = G(g ⊙ (E x) + b)
D = (AᵀA + α_eff I)⁻¹ AᵀY
y = Dᵀa
```

The practical consequence is that the expensive part of training is not
backpropagation through all weights, but accumulation of the sufficient
statistics `AᵀA` and `AᵀY`.  The same primitive supports:

- single-layer supervised learning with `NEFLayer`
- ensembles with `NEFEnsemble`
- online / continuous updates via `partial_fit`, `solve_accumulated`,
  and Woodbury-style inverse updates
- temporal classification with `StreamingNEFClassifier`
- multi-layer and recurrent variants for controlled comparisons
- reinforcement-learning agents built on the same analytic solve

The current supervised-learning paper is
[`docs/SSUP-I.md`](docs/SSUP-I.md).  It is the authoritative source for
the repository's present claims, numbers, and caveats.

### What the default supervised cell is

Paper I's default image-classification cell is:

- `abs` activation
- `hypersphere` encoders
- per-neuron gain `U(0.5, 2.0)`
- data-driven biases via `centers=x_train`
- trace-scaled Tikhonov regularization

That default is strong on raw MNIST / Fashion-MNIST, but the paper's
main empirical message is not "one default beats everything."  The
important levers are encoder choice, bias placement, and regularization
scale.

## Quick start

```python
from leenef.layers import NEFLayer
from leenef.ensemble import NEFEnsemble
from leenef.networks import NEFNetwork

# Single layer: fixed encoders, analytic decoder solve
layer = NEFLayer(d_in=784, n_neurons=5000, d_out=10, centers=x_train)
layer.fit(x_train, y_train)
predictions = layer(x_test)

# Incremental / online accumulation
online = NEFLayer(d_in=784, n_neurons=5000, d_out=10, centers=x_train)
for batch_x, batch_y in data_stream:
    online.partial_fit(batch_x, batch_y)
online.solve_accumulated(alpha=1e-3)

# Ensemble of independent feature banks
ens = NEFEnsemble(
    d_in=784,
    n_neurons=3000,
    d_out=10,
    n_members=10,
    encoder_strategy="receptive_field",
    encoder_kwargs={"image_shape": (28, 28)},
    centers=x_train,
)
ens.fit(x_train, y_train)

# Multi-layer comparison model with six training strategies
net = NEFNetwork(d_in=784, d_out=10, hidden_neurons=[1000], output_neurons=2000)
net.fit_hybrid_e2e(x_train, y_train)
```

## Setup

Requires Python 3.12+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

## Tests

```bash
source venv/bin/activate
pytest
pytest -k test_fit_identity -q
ruff check src/leenef/ tests/ benchmarks/
ruff format --check src/leenef/ tests/ benchmarks/
```

## Benchmarks and paper artifacts

The active supervised-learning harness is the `sl_clean` stack used for
Paper I.  Example launches:

```bash
source venv/bin/activate

PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_mnist
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_fashion
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_cifar10

# Follow-up bias-placement panels
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_cifar10_centered
PYTHON=$PWD/venv/bin/python SEEDS=0-9 \
    bash benchmarks/sl_clean/sl_hp_driver.sh biases_mnist_decentered

# Regenerate the paper figures
python3 docs/generate_paper_i_figures.py
```

All paper-grade result rows carry a code identity, dataset hashes, full
config dictionaries, per-seed metrics, and aggregate summaries.  The
preserved JSON/log artifacts used by Paper I live in `results/sl_clean/`.

## Paper I summary results

All numbers below are multi-seed results from [`docs/SSUP-I.md`](docs/SSUP-I.md).

| Setting | Result | Notes |
|---|---|---|
| Single-layer leenef, MNIST raw | 97.08 ± 0.04 % | hypersphere encoders, `n = 5000` |
| Single-layer leenef, Fashion-MNIST raw | 87.74 ± 0.06 % | hypersphere encoders, `n = 5000` |
| Single-layer leenef, CIFAR-10 raw | 49.72 ± 0.08 % | hypersphere encoders, `n = 5000` |
| Single-layer leenef, CIFAR-10 receptive-field | **55.10 ± 0.10 %** | +1.37 pp over the tuned MLP cell, ~11x less end-to-end wall-clock time |
| Stream-NEF, row sMNIST | **98.45 ± 0.02 %** | `n = 8000`; 0.6 pp below a tuned near-best LSTM, 2.4x less wall-clock time |
| Stream-NEF, permuted sMNIST | **79.44 ± 0.03 %** | `n = 16000`; ESN reaches matched accuracy with fewer parameters |

The corresponding negative results matter just as much:

- generic fixed encoders are not a substitute for spatial priors on
  harder image data
- purely analytic multi-layer stacking does not buy depth on its own
- mean-pooled delay-line features lose the long-range order needed for
  harder sequence tasks

## Components

| Path | Purpose |
|---|---|
| `src/leenef/layers.py` | `NEFLayer`: encode -> activate -> decode; batch, accumulated, and continuous analytic solves |
| `src/leenef/ensemble.py` | `NEFEnsemble`: independent NEF layers combined by averaging or voting |
| `src/leenef/networks.py` | `NEFNetwork`: multi-layer comparison model with greedy, hybrid, target-prop, end-to-end, hybrid->E2E, and target-prop->E2E training |
| `src/leenef/recurrent.py` | `RecurrentNEFLayer`: recurrent decode-then-re-encode sequence model |
| `src/leenef/streaming.py` | `StreamingNEFClassifier`: delay-line temporal classifier with batch, Woodbury, and accumulate+solve training |
| `src/leenef/conv.py` | `ConvNEFStage`, `ConvNEFPipeline`, `ConvNEFEnsemble`: gradient-free convolutional front ends and ensembles |
| `src/leenef/rl.py` | `NEFFeatures`, `NEFFQIAgent`, `NEFFQIEnsemble`: RL agents and ensembles built on analytic value solves |
| `src/leenef/encoders.py` | Encoder strategies: `hypersphere`, `gaussian`, `sparse`, `receptive_field`, `whitened`, `class_contrast`, `local_pca` |
| `src/leenef/activations.py` | Activations: `abs`, `relu`, `softplus`, `lif_rate` |
| `src/leenef/solvers.py` | Decoder solvers and normal-equation utilities |
| `benchmarks/sl_clean/` | Active reproducible supervised-learning sweep harness for Paper I |
| `results/sl_clean/` | Preserved multi-seed result rows, summaries, and logs used by Paper I |
| `attic/` | Archived pre-restart infrastructure, reports, and notes kept for historical reference |
