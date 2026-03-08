# AGENTS.md — leenef

## Commands

```bash
source venv/bin/activate

# Tests
pytest                                        # full suite
pytest tests/test_core.py -q                  # core module tests
pytest tests/test_core.py::TestNEFLayer -q    # single test class
pytest -k test_fit_identity -q                # single test by name

# Install after changing pyproject.toml
pip install -e '.[dev]'
```

## Architecture

This project implements the Neural Engineering Framework (NEF) of Eliasmith
and Anderson for
supervised learning using rate-based neurons on top of PyTorch.

A **NEF layer** has three stages:

1. **Encode** — random unit encoders project the input into a
   high-dimensional neuron space: `a = activation(gain * ((x − d) · e))`
   where *e* is a random direction and *d* is a reference point
   (center) sampled from training data.
2. **Activate** — a nonlinear activation (default: `abs`) models the
   neuron firing rate.
3. **Decode** — output weights (decoders) map activities to the target:
   `y = a @ D`.

Biases are derived from centers: `bias = −gain · (d · e)`.  Default
configuration: **abs** activation, **hypersphere** encoders, **data-driven
biases** via `centers=x_train`.

The key insight: **encoders are random and fixed; decoders are solved
analytically** via regularised least-squares (`layer.fit(x, targets)`).
This avoids gradient-based training for a single layer entirely.

For multi-layer networks (`NEFNetwork` in `networks.py`), hidden layers
use encode-only (activities as inter-layer representation) and only the
output layer decodes.  Three training strategies are supported:

- **Greedy** (`fit_greedy`) — random hidden encoders, analytic output
  decoders.  Fastest, no gradient computation.
- **Hybrid** (`fit_hybrid`) — alternate analytic decoder solves with
  gradient updates to all encoders/biases.
- **End-to-end** (`fit_end_to_end`) — standard SGD on all parameters,
  initialised via a greedy NEF solve.

### Module roles

- `encoders.py` — encoder generation strategies, each registered in
  `ENCODER_STRATEGIES` dict.  Use `make_encoders(n, dim, strategy=...)`.
- `activations.py` — rate neuron models, registered in `ACTIVATIONS` dict.
  Use `make_activation(name, ...)`.
- `solvers.py` — decoder solvers, registered in `SOLVERS` dict.
  Use `solve_decoders(A, targets, method=...)`.
- `layers.py` — `NEFLayer(nn.Module)` ties them together.
  `forward()` runs the full pipeline; `fit()` solves decoders analytically.
  Accepts optional `centers=` training data to derive data-driven biases
  (`bias = -gain * (d · e)`), placing each neuron around a training sample.
- `networks.py` — `NEFNetwork(nn.Module)` stacks NEFLayers with the three
  training strategies above.

## Conventions

### Registry pattern

Encoders, activations, and solvers each use a `dict` registry
(`ENCODER_STRATEGIES`, `ACTIVATIONS`, `SOLVERS`) keyed by string name, with
a `make_*` / `solve_*` factory function.  New variants should be added to
the registry dict; callers select by string name.

### Decoders as `nn.Parameter`

Decoders are stored as `nn.Parameter(requires_grad=False)` so they
participate in `state_dict` and model saving, but are not updated by
gradient optimisers by default.  `fit()` writes to `.data` directly.

### Tensor-only, no Python loops

All computation must use batched PyTorch tensor operations
(`torch.linalg`, `@`, broadcasting).  Never loop over neurons or samples
in Python.
