---
name: leenef-method-selection
description: Choose, configure, and reason about leenef fixed-feature analytic-solve methods for supervised, temporal, continual, and reinforcement-learning tasks, plus general guidance on the package's APIs, defaults, benchmarks, and reproducibility conventions. Use when a user asks about leenef components, encoders, alpha policies, neuron counts, solve paths, GPU strategies, benchmark configurations, or related advice.
when_to_use: >
  Invoke this skill before writing leenef code, configuring experiments, porting
  a task to leenef, explaining why a leenef method does or does not fit a
  problem, or advising on benchmarks, defaults, or reproducibility. Prefer
  source- and paper-grounded recommendations over canned defaults.
---

# leenef method-selection skill

Help the user choose a leenef configuration by reading the task, expected data
shape, target semantics, hardware, and tolerance for feature limitations.  Do
not present leenef as a universal replacement for gradient training.  The core
primitive is fixed activities `A`, accumulated sufficient statistics `A^T A`
and `A^T Y`, and an analytic decoder solve.

This file is self-contained.  To install it in an agent that follows the
Agent Skills convention, copy this whole skill directory (this file plus the
adjacent `docs/`, `src/`, `benchmarks/`, `tests/`, `colab/`, and `results/`
folders) into the agent's skill directory.  If you installed leenef from
PyPI, run `leenef-install-skill --dest <skills-dir>` to materialize that
directory for you (see `docs/USAGE.md`).

## Always read these before recommending

The skill ships with the package and is fully self-contained — every file
referenced below is available at the relative path shown, including after a
plain `pip install leenef`.  Do not recommend a configuration from memory;
read the relevant material first.

Start here:

- [`docs/USAGE.md`](docs/USAGE.md) — user-facing configuration guide.
- [`README.md`](README.md) — project overview, paper abstracts, component map.
- [`AGENTS.md`](AGENTS.md) — operational conventions (reproducibility hazards,
  alpha-policy rules, naming, commit style); read before suggesting code edits.

Then read the SSUP paper that matches the task family:

- [`docs/SSUP-I.md`](docs/SSUP-I.md) — supervised/static/streaming evidence and
  alpha guidance.
- [`docs/SSUP-II.md`](docs/SSUP-II.md) — continual-learning additivity, CL limits.
- [`docs/SSUP-III.md`](docs/SSUP-III.md) — RL target-construction evidence,
  RL limits, NEF-AVR / NEF-SF caveats.

Reach for these when the task touches the corresponding area:

- `docs/sl_clean_sweeps.md`, `docs/cl_clean_sweeps.md`, `docs/rl_clean_sweeps.md`,
  `docs/rl_sparse_sweeps.md` — chronological sweep logs and verdicts.
- `docs/cl_clean_notes.md`, `docs/cl_clean_handover.md`,
  `docs/rl_sparse_notes.md`, `docs/rl_paper_iii_notes.md` — synthesized
  research notes.
- `src/leenef/` — exact public APIs and defaults
  (`layers.py`, `ensemble.py`, `networks.py`, `recurrent.py`, `streaming.py`,
  `conv.py`, `rl.py`, `rl_sf.py`, `rl_diag.py`, `encoders.py`, `activations.py`,
  `solvers.py`).  Quote the actual signature when recommending a constructor.
- `benchmarks/` — reference drivers (`sl_clean/`, `cl_clean/`, `run_rl_paper_dense.py`,
  `run_rl_sparse.py`, etc.) and their `*.sh` orchestrators.  Mirror these
  patterns for new experiments.
- `tests/` — executable usage examples and invariants (`test_layers.py`,
  `test_rl.py`, `test_streaming.py`, ...).
- `results/**/base_config*.json` — paper-grade default configs adopted by the
  multi-seed sweep protocol.  Use these as starting points, not memory.
- `colab/` — GPU notebooks and runners for large-scale variants.

## Decision procedure

1. Classify the task:
   - static supervised vector/tabular data -> `NEFLayer`;
   - static image data -> `NEFLayer` with receptive-field encoders, or
     `ConvNEFPipeline` for harder visual tasks;
   - sequence classification -> `StreamingNEFClassifier`;
   - continual/task streams with stable output semantics -> accumulated
     `NEFLayer` statistics;
   - dense discrete-action RL -> `NEFAVRAgent`;
   - sparse-reward RL research -> `NEFSFAgent`.
2. Ask whether target semantics are stable.  If output columns change meaning,
   do not promise the Paper-II no-forgetting identity.
3. Ask whether fixed features are likely sufficient.  If visual/spatial
   structure matters, prefer receptive-field or ConvNEF front ends before only
   increasing `n_neurons`.
4. Ask whether the run is large enough to need GPU/chunking.  For large
   activity matrices prefer accumulate-and-solve paths over Woodbury inverse
   maintenance.
5. Give a starting config plus the main caveat and the first tuning sweep.
   Cite a specific `base_config*.json` and/or driver under `benchmarks/` when
   one applies.

## Conservative starting configurations

| Situation | Starting point |
|---|---|
| Static vector/tabular supervised | `NEFLayer(..., n_neurons=1000-5000, activation="abs", encoder_strategy="hypersphere", gain=(0.5, 2.0), centers=x_train)` with `alpha=(1e-3, "trace")` for SSUP-static-like panels or `alpha=(1e-2, "trace")` as the API default. |
| Static image vectors | `NEFLayer(..., n_neurons=5000, encoder_strategy="receptive_field", encoder_kwargs={"image_shape": (H, W)}, centers=x_train)`, tune trace alpha over `1e-3` and `1e-2`. |
| Hard visual tasks | `ConvNEFPipeline` with standardized features and a NEF head. Use before claiming generic encoders are weak. |
| Ensembles | `NEFEnsemble` with 5-20 independent members and probability averaging. |
| Sequence classification | `StreamingNEFClassifier(..., window_size=10, n_neurons=4000-8000, centers=x_train_seq)`, `alpha=(1e-4, "trace")`, `accumulate(..., chunk_stats=True)`, then `solve()`. |
| Continual class/task streams | Accumulate with `partial_fit` per task/chunk, keep output columns stable, use `n_neurons=5000`, representative or first-task centers, `alpha=(1e-3, "trace")`, then `solve_accumulated()`. |
| Streaming continual sequences | Streaming classifier with chunked accumulation, `window_size=10`, `n_neurons=4000+`, `alpha=(1e-4, "trace")`, GPU if available. |
| Dense RL | `NEFAVRAgent(target_mode="nstep", n_step=100, n_neurons=6000, gamma=0.98, alpha=0.1, forget_factor=0.999, solve_every=10)`. Start with stable non-bootstrapped targets; do not start from one-step TD. |
| Sparse RL | `NEFSFAgent` with an explicit goal/hindsight/successor target. Start at `n_neurons=2000`; use `device="cuda"` for full-successor matrices when available. Width alone is not expected to fix sparse rewards. |

## Alpha and solve-path rules

- Normal-equation APIs should receive explicit trace policies such as
  `alpha=(1e-2, "trace")`, `alpha=(1e-3, "trace")`, or for streaming
  `alpha=(1e-4, "trace")`.
- Fixed-alpha Woodbury APIs use `fixed_alpha=...`; do not change it while a
  cached inverse exists unless the code calls `refresh_inverse()` or
  `reset_continuous()` first.
- For large GPU runs, prefer `accumulate(..., chunk_stats=True)` plus `solve()`
  to avoid storing and updating a dense inverse.

## Recommendation style

Return:

1. the chosen leenef component;
2. a compact starting configuration;
3. why that choice matches the task;
4. the likely limitation;
5. the first small sweep or validation check.

If the problem is outside the evidence base, say so plainly and frame the
configuration as an experiment rather than a supported default.
