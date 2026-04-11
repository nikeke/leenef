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

# Report figures
python docs/generate_figures.py

# Benchmarks (use --seed 0 so reruns stay comparable)
python benchmarks/run.py --multi --seed 0
python benchmarks/run.py --ensemble --ensemble-members 20 --ensemble-receptive-field --seed 0
python benchmarks/run_recurrent.py --seed 0
python benchmarks/run_recurrent.py --streaming --streaming-neurons 4000 --streaming-window 10 --seed 0
python benchmarks/run_recurrent.py --streaming --streaming-solve-mode accumulate --streaming-neurons 4000 --streaming-window 10 --seed 0

# Colab suites (GPU)
python benchmarks/colab_suites.py --suite row_focus --device auto --output-dir results/colab
python benchmarks/colab_suites.py --suite sequential_hard --device auto --output-dir results/colab
python benchmarks/colab_suites.py --suite conv_cifar --device auto --output-dir results/colab
```

## Technical report maintenance

The technical report (TR) is `docs/technical_report.md`. Treat it as a
globally coupled document: later edits often change earlier framing,
summary tables, figure captions, and even the title.

### Primary files

- `docs/technical_report.md` — main report text
- `docs/generate_figures.py` — authoritative figure data, labels, and plot
  text
- `docs/figures/*.png` — generated assets; regenerate them whenever figure
  data, labels, or captions change

### Narrative spine and cross-section propagation

- The **title, abstract, Section 1, Section 6, and Section 7** are the
  narrative spine. When headline results, defaults, or claims change,
  revisit all of them, not just the local section you edited.
- Do not stop at the directly touched paragraph or table. Propagate changes
  through:
  - table highlights
  - captions and figure references
  - cross-references between sections
  - summary/recommendation tables
  - comparison prose (`beats`, `matches`, `almost matches`, etc.)
  - README or other docs if they repeat the same claim
- After a major results edit, do a final top-down coherence pass over the
  abstract, introduction, conclusions, and title together.

### Style and numeric conventions

- Use **American English everywhere** in the report, documentation, and
  code-facing prose. Common conversions:
  - `normalise` / `normalisation` -> `normalize` / `normalization`
  - `initialise` / `initialisation` -> `initialize` / `initialization`
  - `optimise` / `optimisation` -> `optimize` / `optimization`
  - `regularise` / `regularised` -> `regularize` / `regularized`
  - `colour` -> `color`
  - `behaviour` -> `behavior`
  - `modelling` -> `modeling`
  - `neighbourhood` -> `neighborhood`
- Round this project's reported **percentages to one decimal place**
  everywhere in the TR:
  - prose
  - tables
  - table highlights
  - figure labels and legends
  - captions
  - summary bullets
- **Exceptions:** keep externally cited literature values at the precision
  used in the source paper (for example McDonnell et al.'s `99.17%`), and
  keep raw console/log output unrounded when reproducing it verbatim.
- After rounding, re-check the meaning of every comparison:
  - ties created by rounding must be highlighted equally
  - a previous "better than" may become "matches"
  - a previous "matches" may need to become "almost matches"
  - if two rows round to the same result and one is slower or otherwise
    dominated, usually remove or de-emphasize the weaker row
- Do not use space-separated thousands in the TR. Use `12000`, `20000`,
  `60000`, not `12 000`, `20 000`, `60 000`.
- Keep hardware qualification explicit. Default phrasing: **results are on
  CPU unless stated otherwise**. If speed is a headline point and both CPU
  and GPU measurements exist, mention both. Do not merge CPU accuracy and
  GPU timing into a single unqualified claim.

### Content placement and structure

- Background sections should stay conceptual. Implementation-specific
  paragraphs belong in methods/implementation sections, not in conceptual
  background.
- Benchmark tables and quantitative comparisons belong in results or
  discussion sections, not in method descriptions.
- When moving content between sections, update all references to it.
- Thin empirical sections should usually be strengthened with a familiar
  baseline or comparison, not left as isolated internal numbers.
- Summary tables should avoid duplicate scenarios and keep columns
  semantically consistent across rows.
- Check table highlighting carefully: every rounded tie for the best value
  should be highlighted, and no highlight should imply unsupported
  superiority.

### Code/report alignment

- Keep the TR synchronized with actual code defaults and capabilities:
  - 7 encoder strategies (`hypersphere`, `gaussian`, `sparse`,
    `receptive_field`, `whitened`, `class_contrast`, `local_pca`)
  - single-layer default activation `abs`
  - recurrent default activation `relu`
  - data-driven biases via `centers=...`
  - GPU-friendly accumulate+solve path in addition to Woodbury updates
- Avoid blanket "random" language. In this project, encoders are **fixed**
  but may be random or data-adapted; biases may be data-driven. Use
  mechanism-specific wording instead of oversimplifying.
- If figure data changes, update `docs/generate_figures.py`, regenerate the
  PNGs, then update in-text references and captions in the TR.
- If a headline claim changes, re-check whether the title still fits the
  revised abstract/introduction/conclusions.

## Architecture

This project implements the Neural Engineering Framework (NEF) of Eliasmith
and Anderson for
supervised learning using rate-based neurons on top of PyTorch.

A **NEF layer** has three stages:

1. **Encode** — fixed unit encoders project the input into a
   high-dimensional neuron space: `a = activation(gain * ((x − d) · e))`
   where *e* is a fixed direction (often random, but potentially
   data-adapted) and *d* is a reference point (center) sampled from
   training data.
2. **Activate** — a nonlinear activation (default: `abs`) models the
   neuron firing rate.
3. **Decode** — output weights (decoders) map activities to the target:
   `y = a @ D`.

Biases are derived from centers: `bias = −gain · (d · e)`.  Default
configuration: **abs** activation, **hypersphere** encoders, **per-neuron
gain** U(0.5, 2.0), **data-driven biases** via `centers=x_train`.
Recurrent layers default to **relu** (abs causes gradient explosion in BPTT).

The key insight: **encoders are fixed (random or data-adapted); decoders
are solved analytically** via regularized least-squares
(`layer.fit(x, targets)`). This avoids gradient-based training for a
single layer entirely.

For multi-layer networks (`NEFNetwork` in `networks.py`), hidden layers
use encode-only (activities as inter-layer representation) and only the
output layer decodes.  Five training strategies are supported:

- **Greedy** (`fit_greedy`) — fixed hidden encoders, analytic output
  decoders.  Fastest, no gradient computation.
- **Hybrid** (`fit_hybrid`) — alternate analytic decoder solves with
  gradient updates to all encoders/biases.
- **Target propagation** (`fit_target_prop`) — replaces backprop with
  layer-local targets via analytical representational decoders (NEF
  inverse models) and difference target propagation.  Single-layer
  gradients only; no gradient flows between layers.
- **End-to-end** (`fit_end_to_end`) — standard SGD on all parameters,
  initialized via a greedy NEF solve.

### Module roles

- `encoders.py` — encoder generation strategies, each registered in
  `ENCODER_STRATEGIES` dict.  Use `make_encoders(n, dim, strategy=...)`.
  Strategies: `hypersphere` (default), `gaussian`, `sparse`,
  `receptive_field` (local image patches for spatial locality),
  `whitened` (PCA-subspace projection adapts to data covariance),
  `class_contrast` (directions from one class to nearest other class),
  `local_pca` (top eigenvector of each neuron's local neighborhood).
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
  materializing the full T×B activity matrix during greedy solve.
  Supports data-driven biases via `centers=` (first timestep + zero
  state).
- `streaming.py` — `StreamingNEFClassifier(nn.Module)` classifies
  variable-length temporal sequences using a delay-line reservoir approach.
  Overlapping windows of K consecutive timesteps are encoded through fixed
  NEF neurons, mean-pooled over time, and decoded to class labels.
  Supports batch `fit()`, continuous Woodbury `continuous_fit()`, and
  GPU-friendly `accumulate()` + `solve()` (float32, no Woodbury inverse).
  `encode_sequence()` chunks internally to limit peak memory for large
  models.  Achieves 98.6% on sMNIST-row without gradient descent.
- `conv.py` — gradient-free convolutional feature extraction pipeline.
  `ConvNEFStage` learns PCA (or k-means) filters from data patches and
  applies them as fixed conv2d filters + abs activation + pool.
  `patch_size` accepts int (square) or `(h, w)` tuple (rectangular) for
  oriented filters.  `ConvNEFPipeline` stacks stages (sequential or
  parallel) with spatial pyramid pooling and a NEFLayer classification
  head.  Supports feature standardization (`standardize=True`), global
  contrast normalization (`gcn=True`), local contrast normalization
  (`lcn_kernel=5`), and data augmentation (`augment_fn`, `n_augment`).
  `parallel=True` enables multi-scale extraction with different patch
  sizes concatenated along the channel dimension.  `ConvNEFEnsemble`
  wraps N pipelines with different seeds.  All gradient-free: PCA
  filters are data-derived buffers, decoders are solved analytically
  via partial_fit/solve_accumulated.

## Conventions

### Registry pattern

Encoders, activations, and solvers each use a `dict` registry
(`ENCODER_STRATEGIES`, `ACTIVATIONS`, `SOLVERS`) keyed by string name, with
a `make_*` / `solve_*` factory function.  New variants should be added to
the registry dict; callers select by string name.

### Decoders as `nn.Parameter`

Decoders are stored as `nn.Parameter(requires_grad=False)` so they
participate in `state_dict` and model saving, but are not updated by
gradient optimizers by default.  `fit()` writes to `.data` directly.

### Tensor-only, no Python loops

All computation must use batched PyTorch tensor operations
(`torch.linalg`, `@`, broadcasting).  Never loop over neurons or samples
in Python.
