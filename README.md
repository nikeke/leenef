# leenef

Supervised learning experiments with Eliasmith's Neural Engineering Framework
(NEF), using rate-based neurons on PyTorch.

## Overview

Instead of training neural network weights with gradient descent, NEF
computes optimal output weights (decoders) analytically via regularised
least-squares.  Input weights (encoders) are random and fixed.  This is
equivalent to a random-feature model derived from neuroscience principles.

The library provides a `NEFLayer` that plugs into standard PyTorch workflows:

```python
from leenef.layers import NEFLayer

layer = NEFLayer(d_in=784, n_neurons=2000, d_out=10)
layer.fit(x_train, y_train)       # analytic solve — no epochs, no optimizer
predictions = layer(x_test)
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
pytest                                     # full suite
pytest -k test_fit_identity -q             # single test
```

## Benchmarks

Run the benchmark suite:

```bash
python benchmarks/run.py --datasets mnist fashion_mnist cifar10 \
       --neurons 500 1000 2000 5000 --regression
```

### Single-layer results (2000 neurons, Tikhonov solver, α = 0.01)

#### Scaling with neuron count (ReLU + hypersphere)

| Dataset        |  500   | 1000   | 2000   | 5000   |
|----------------|--------|--------|--------|--------|
| MNIST          | 88.4%  | 89.9%  | 92.1%  | 94.2%  |
| Fashion-MNIST  | 80.6%  | 82.0%  | 83.3%  | 84.5%  |
| CIFAR-10       | 40.8%  | 42.8%  | 44.8%  | 46.7%  |

#### Encoder × activation (2000 neurons, test accuracy)

|              | hypersphere | gaussian | sparse |
|--------------|-------------|----------|--------|
| **MNIST**    |             |          |        |
| relu         | 91.8%       | 95.6%    | 95.6%  |
| softplus     | 87.7%       | **95.9%**| 95.7%  |
| lif_rate     | 91.3%       | 95.5%    | 95.3%  |
| **Fashion**  |             |          |        |
| relu         | 83.2%       | 85.8%    | 85.6%  |
| softplus     | 81.4%       | **85.9%**| 85.6%  |
| lif_rate     | 83.0%       | 85.8%    | 85.5%  |

#### Regression — California Housing (MSE, normalised targets)

| Neurons | Train MSE | Test MSE |
|---------|-----------|----------|
| 500     | 0.270     | 0.287    |
| 1000    | 0.262     | 0.250    |
| 2000    | 0.246     | 0.240    |
| 5000    | 0.226     | 0.228    |

**Key findings:**
- Encoder distribution has far more impact than activation function.
  Gaussian and sparse encoders outperform hypersphere by 3–8%.
- Softplus + gaussian is the best overall combination (~95.9% MNIST).
- Performance scales monotonically with neuron count.
- CIFAR-10 is limited (~47%) by the single-layer architecture on 3072-d input.
- Fit time is under 12s for 5000 neurons on 60k samples (CPU).

## Components

| Module | Purpose |
|--------|---------|
| `leenef/encoders.py` | Random encoder generation (hypersphere, Gaussian, sparse) |
| `leenef/activations.py` | Rate-based neuron models (ReLU, softplus, LIF rate) |
| `leenef/solvers.py` | Decoder solvers (lstsq, Tikhonov, Cholesky) |
| `leenef/layers.py` | `NEFLayer(nn.Module)` — encode → activate → decode |
| `benchmarks/run.py` | Benchmark harness with timing and accuracy reporting |
