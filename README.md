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

## Components

| Module | Purpose |
|--------|---------|
| `leenef/encoders.py` | Random encoder generation (hypersphere, Gaussian, sparse) |
| `leenef/activations.py` | Rate-based neuron models (ReLU, softplus, LIF rate) |
| `leenef/solvers.py` | Decoder solvers (lstsq, Tikhonov, Cholesky) |
| `leenef/layers.py` | `NEFLayer(nn.Module)` — encode → activate → decode |
