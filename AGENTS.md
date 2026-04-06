# AGENTS.md — leenef

## Commands

```bash
source venv/bin/activate

# Tests
pytest                                        # full suite
pytest tests/test_core.py -q                  # core module tests
pytest tests/test_core.py::TestNEFLayer -q    # single test class
pytest -k test_fit_identity -q                # single test by name

# Lint
ruff check src/leenef/ tests/ benchmarks/        # lint (E/F/W/I rules)
ruff format --check src/leenef/ tests/ benchmarks/  # format check

# Install after changing pyproject.toml
pip install -e '.[dev]'

# Benchmarks (use --seed 0 so reruns stay comparable)
python benchmarks/run.py --multi --seed 0
python benchmarks/run.py --ensemble --ensemble-members 20 --ensemble-receptive-field --seed 0
python benchmarks/run_recurrent.py --seed 0
python benchmarks/run_recurrent.py --streaming --streaming-neurons 4000 --streaming-window 10 --seed 0
python benchmarks/run_recurrent.py --streaming --streaming-solve-mode accumulate --streaming-neurons 4000 --streaming-window 10 --seed 0
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
configuration: **abs** activation, **hypersphere** encoders, **per-neuron
gain** U(0.5, 2.0), **data-driven biases** via `centers=x_train`.
Recurrent layers default to **relu** (abs causes gradient explosion in BPTT).

The key insight: **encoders are random and fixed; decoders are solved
analytically** via regularised least-squares (`layer.fit(x, targets)`).
This avoids gradient-based training for a single layer entirely.

For multi-layer networks (`NEFNetwork` in `networks.py`), hidden layers
use encode-only (activities as inter-layer representation) and only the
output layer decodes.  Five training strategies are supported:

- **Greedy** (`fit_greedy`) — random hidden encoders, analytic output
  decoders.  Fastest, no gradient computation.
- **Hybrid** (`fit_hybrid`) — alternate analytic decoder solves with
  gradient updates to all encoders/biases.
- **Target propagation** (`fit_target_prop`) — replaces backprop with
  layer-local targets via analytical representational decoders (NEF
  inverse models) and difference target propagation.  Single-layer
  gradients only; no gradient flows between layers.
- **End-to-end** (`fit_end_to_end`) — standard SGD on all parameters,
  initialised via a greedy NEF solve.

### Module roles

- `encoders.py` — encoder generation strategies, each registered in
  `ENCODER_STRATEGIES` dict.  Use `make_encoders(n, dim, strategy=...)`.
  Strategies: `hypersphere` (default), `gaussian`, `sparse`,
  `receptive_field` (local image patches for spatial locality).
- `activations.py` — rate neuron models, registered in `ACTIVATIONS` dict.
  Use `make_activation(name, ...)`.
- `solvers.py` — decoder solvers, registered in `SOLVERS` dict.
  Use `solve_decoders(A, targets, method=...)`.
- `layers.py` — `NEFLayer(nn.Module)` ties them together.
  `forward()` runs the full pipeline; `fit()` solves decoders analytically.
  `partial_fit()` / `solve_accumulated()` enable incremental/online learning
  via normal-equation accumulation.
  `continuous_fit()` / `continuous_fit_encoded()` apply rank-k Woodbury
  updates to the cached system inverse, producing up-to-date decoders
  after every batch.  The inverse is stored in float64 to prevent drift.
  `refresh_inverse()` recomputes the inverse exactly from accumulated
  statistics; `reset_continuous()` clears all Woodbury state.
  Accepts optional `centers=` training data to derive data-driven biases
  (`bias = -gain * (d · e)`), placing each neuron around a training sample.
  Gain can be scalar, range tuple, or per-neuron tensor (`_gain` buffer).
  `encoder_kwargs=` dict passes strategy-specific args (e.g. `image_shape`).
- `ensemble.py` — `NEFEnsemble(nn.Module)` wraps N independent NEFLayers
  with different random seeds.  Combines via probability averaging (`mean`)
  or majority voting (`vote`).  Ideal for exploiting the fast analytic
  solve: 20 members × 2s = 40s, still faster than one gradient-trained MLP.
- `networks.py` — `NEFNetwork(nn.Module)` stacks NEFLayers with the five
  training strategies above.  Also supports `hybrid_e2e` (hybrid → E2E).
- `recurrent.py` — `RecurrentNEFLayer(nn.Module)` implements the NEF
  decode-then-re-encode feedback loop for temporal sequences.  State
  decoders close the recurrent loop; output decoders produce the task
  prediction at the final timestep.  Four strategies (greedy/hybrid/
  target-prop/E2E); only E2E produces competitive results — greedy and
  hybrid struggle because random encoders compound state feedback noise
  across timesteps.  Target propagation through time (TPTT) uses state
  decoders as inverse models to propagate targets backward through
  timesteps.  `solve_from_normal_equations` in `solvers.py` avoids
  materialising the full T×B activity matrix during greedy solve.
  Supports data-driven biases via `centers=` (first timestep + zero
  state).
- `streaming.py` — `StreamingNEFClassifier(nn.Module)` classifies
  variable-length temporal sequences using a delay-line reservoir approach.
  Overlapping windows of K consecutive timesteps are encoded through random
  NEF neurons, mean-pooled over time, and decoded to class labels.
  Supports batch `fit()`, continuous Woodbury `continuous_fit()`, and
  GPU-friendly `accumulate()` + `solve()` (float32, no Woodbury inverse).
  `encode_sequence()` chunks internally to limit peak memory for large
  models.  Achieves 98.57% on sMNIST-row without gradient descent.

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
