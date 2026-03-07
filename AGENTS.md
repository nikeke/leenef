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

This project implements Eliasmith's Neural Engineering Framework (NEF) for
supervised learning using rate-based neurons on top of PyTorch.

A **NEF layer** has three stages:

1. **Encode** — random input weights (encoders) project the input into a
   high-dimensional neuron space: `a = activation(gain * (x @ E^T) + bias)`.
2. **Activate** — a nonlinear activation models the neuron firing rate.
3. **Decode** — output weights (decoders) map activities to the target:
   `y = a @ D`.

The key insight: **encoders are random and fixed; decoders are solved
analytically** via regularised least-squares (`layer.fit(x, targets)`).
This avoids gradient-based training for a single layer entirely.

For multi-layer networks (planned), encoders can optionally be unfrozen
(`trainable_encoders=True`) and trained with backprop while decoders are
re-solved analytically after each encoder update.

### Module roles

- `encoders.py` — encoder generation strategies, each registered in
  `ENCODER_STRATEGIES` dict.  Use `make_encoders(n, dim, strategy=...)`.
- `activations.py` — rate neuron models, registered in `ACTIVATIONS` dict.
  Use `make_activation(name, ...)`.
- `solvers.py` — decoder solvers, registered in `SOLVERS` dict.
  Use `solve_decoders(A, targets, method=...)`.
- `layers.py` — `NEFLayer(nn.Module)` ties them together.
  `forward()` runs the full pipeline; `fit()` solves decoders analytically.

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
