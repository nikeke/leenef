# leenef

Supervised learning with the Neural Engineering Framework (NEF) of
Eliasmith and Anderson, using rate-based neurons on PyTorch.

**[Technical report](docs/technical_report.md)** — full background, method
details, and benchmark analysis.

## Overview

A standard neural network trains all weights with gradient descent.  NEF
takes a different approach: input weights (encoders) are random and fixed,
and output weights (decoders) are solved analytically via regularised
least-squares.  No gradient descent, no epochs, no learning rate — a single
layer trains in under 2 seconds on a laptop CPU.

Each neuron computes `activity = |gain · ((x − d) · e)|`, where **e** is a
random unit direction, **d** is a reference point sampled from training
data, and **gain** controls sensitivity.  The neuron measures unsigned
deviation from a known reference along a random direction.

The library provides:
- `NEFLayer` — single-layer analytical solve
- `NEFEnsemble` — ensemble of independent NEFLayers
- `NEFNetwork` — multi-layer with six training strategies
- `RecurrentNEFLayer` — temporal sequence models

## Quick start

```python
from leenef.layers import NEFLayer
from leenef.ensemble import NEFEnsemble

# Single layer — 2 seconds, 95.7% MNIST
layer = NEFLayer(d_in=784, n_neurons=2000, d_out=10, centers=x_train)
layer.fit(x_train, y_train)
predictions = layer(x_test)

# Ensemble with local receptive fields — 28 seconds, 96.5% MNIST
ens = NEFEnsemble(d_in=784, n_neurons=2000, d_out=10,
                  n_members=10, encoder_strategy="receptive_field",
                  encoder_kwargs={"image_shape": (28, 28)},
                  centers=x_train)
ens.fit(x_train, y_train)

# Incremental / online learning
layer = NEFLayer(d_in=784, n_neurons=2000, d_out=10, centers=x_train)
for batch_x, batch_y in data_stream:
    layer.partial_fit(batch_x, batch_y)
layer.solve_accumulated(alpha=1e-2)

# Multi-layer — six training strategies
from leenef.networks import NEFNetwork
net = NEFNetwork(d_in=784, d_out=10, hidden_neurons=[1000],
                 output_neurons=2000, centers=x_train)
net.fit_hybrid_e2e(x, targets)  # best balanced strategy
```

## Setup

Requires Python 3.12+.

The repository uses a standard `src/` layout; the main package lives in
`src/leenef/`.

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

> All timings: CPU-only, AMD Ryzen 5 PRO 5650U, no GPU.

```bash
python benchmarks/run.py --datasets mnist fashion_mnist cifar10 \
       --neurons 500 1000 2000 5000 --regression --seed 0
python benchmarks/run.py --neurons 2000 --multi --mlp --seed 0
python benchmarks/run.py --neurons 2000 --ensemble --ensemble-members 10 \
       --ensemble-receptive-field --seed 0
python benchmarks/run_recurrent.py --mode row --seed 0
```

### Summary results

| Model                     | MNIST  | Fashion | CIFAR-10 | Time (MNIST) |
|---------------------------|--------|---------|----------|--------------|
| NEFLayer (2000)           | 95.7%  | 85.9%   | 47.8%    |     2s       |
| **NEF single RF** (12k, α-tuned) | **98.5%** | — | —   |    **52s**   |
| **NEF single RF** (14k, α-tuned) | — | **89.7%** | —   |    **82s**   |
| **NEF ensemble** (10×3k RF, α-tuned) | — | — | **58.4%** | **53s**  |
| NEFNet-hybrid→E2E         | 98.6%  | 90.6%   | 58.4%    |   402s       |
| MLP (2×1000)              | 98.5%  | 89.7%   | 52.7%    |    83s       |

The single-layer NEF model with local receptive field encoders and tuned
Tikhonov regularisation **matches or beats the gradient-trained MLP on
all three benchmarks while training faster** — without any gradient
computation.  The key insight: at scale (12 000–14 000 RF neurons), the
default regularisation α=10⁻² over-regularises; reducing to α≈5×10⁻⁴
closes the final accuracy gap.  See the
[technical report](docs/technical_report.md) for the full sweep results,
analysis, and competitive context.

## Components

| Module | Purpose |
|--------|---------|
| `src/leenef/layers.py` | `NEFLayer` — encode → activate → decode |
| `src/leenef/ensemble.py` | `NEFEnsemble` — ensemble of NEFLayers |
| `src/leenef/networks.py` | `NEFNetwork` — multi-layer (greedy / hybrid / TP / E2E / hybrid→E2E / TP→E2E) |
| `src/leenef/recurrent.py` | `RecurrentNEFLayer` — temporal decode-then-re-encode loop |
| `src/leenef/encoders.py` | Encoder strategies (hypersphere, Gaussian, sparse, receptive field) |
| `src/leenef/activations.py` | Activations (abs, relu, softplus, lif_rate) |
| `src/leenef/solvers.py` | Decoder solvers (Tikhonov, Cholesky, lstsq, normal equations) |
| `benchmarks/` | Benchmark harnesses and plotting |
| `docs/technical_report.md` | Full technical report |

## References

- C. Eliasmith & C. H. Anderson, *Neural Engineering*, MIT Press, 2003.
- G.-B. Huang et al., "Extreme Learning Machine", *Neurocomputing* 70, 2006.
- A. Rahimi & B. Recht, "Random Features for Large-Scale Kernel Machines", *NeurIPS*, 2007.
- M. D. McDonnell et al., "Fast, Simple and Accurate Handwritten Digit Classification…", *PLOS ONE* 10(8), 2015.

See the [technical report](docs/technical_report.md) for the complete
reference list.
