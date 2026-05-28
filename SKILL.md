---
name: leenef-method-selection
description: Choose and configure leenef fixed-feature analytic-solve methods for supervised, temporal, continual, and reinforcement-learning tasks. Use when a user asks which leenef component, encoder, alpha policy, neuron count, solve path, or GPU strategy to use.
when_to_use: >
  Invoke this skill before writing leenef code, configuring experiments, porting
  a task to leenef, or explaining why a leenef method does or does not fit a
  problem. Prefer documentation- and paper-grounded recommendations over simple
  canned defaults.
---

# leenef method-selection skill

Help the user choose a leenef configuration by reading the task, expected data
shape, target semantics, hardware, and tolerance for feature limitations.  Do
not present leenef as a universal replacement for gradient training.  The core
primitive is fixed activities `A`, accumulated sufficient statistics `A^T A`
and `A^T Y`, and an analytic decoder solve.

This file is self-contained.  To install it in an agent that follows the
Agent Skills convention, place it as `<skill-directory>/SKILL.md`.

## Always check these resources when available

- `docs/USAGE.md` — user-facing configuration guide.
- `docs/SSUP-plain-language.md` — conceptual overview.
- `docs/SSUP-I.md` — supervised/static/streaming evidence and alpha guidance.
- `docs/SSUP-II.md` — continual-learning additivity and CL limits.
- `docs/SSUP-III.md` — RL target-construction evidence and RL limits.
- `src/leenef/` — exact public APIs and defaults.
- `benchmarks/*` and `results/*/base_config*.json` — paper-grade configs.

## Decision procedure

1. Classify the task:
   - static supervised vector/tabular data -> `NEFLayer`;
   - static image data -> `NEFLayer` with receptive-field encoders, or
     `ConvNEFPipeline` for harder visual tasks;
   - sequence classification -> `StreamingNEFClassifier`;
   - continual/task streams with stable output semantics -> accumulated
     `NEFLayer` statistics;
   - dense discrete-action RL -> `NEFFQIAgent`;
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
| Dense RL | `NEFFQIAgent(target_mode="nstep", n_step=100, n_neurons=6000, gamma=0.98, alpha=0.1, forget_factor=0.999, solve_every=10)`. Start with stable non-bootstrapped targets; do not start from one-step TD. |
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
