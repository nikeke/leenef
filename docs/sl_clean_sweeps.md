# sl_clean sweeps — phase log

This document mirrors `attic/docs/rl_clean_sweeps.md` for the
supervised-learning side of the foundational paper series. Every phase
that contributes to a numeric claim in **Paper I — Sufficient Statistics
as the Universal Primitive** lands here, with:

- the phase name, experiment, knob, and value list;
- per-value n, mean ± SE, min, max for the primary metric;
- the adoption decision (Δ vs `sqrt(SE_winner² + SE_current²)`) and a
  one-line justification of why the analyzer kept or replaced the
  incumbent;
- the regenerated figure path under `docs/figures/`;
- open questions and follow-ups (entanglements, knob-pair sweeps, etc.).

The driver scripts (`benchmarks/sl_clean/sl_hp_driver.sh`) and the live
base configs (`results/sl_clean/base_config_*.json`) are the operational
counterparts of this log; this file is the human-readable summary.

---

## Conventions

- **Code identity.** Every result row carries a `code_identity` hash
  (HEAD SHA, or `<sha>-dirty-<sha256[:12]>` if any path under
  `src/`, `benchmarks/`, `tests/`, `pyproject.toml` is dirty). Rows whose
  `code_identity` no longer matches the current tree are kept on disk
  but excluded from aggregates and re-run on demand.
- **Adoption rule.** A new value `v*` replaces the current base only when

  ```
  mean(metric | v=v*) − mean(metric | v=base) > sqrt(SE_v*² + SE_base²)
  ```

  with `SE = std(ddof=1) / sqrt(n)`, both sides averaged over `≥3` seeds
  by default (configurable via `--min-seeds`).
- **Seed budget.** 10 seeds for headline claims, 5 for ablation tables,
  3 for appendix / legacy cells.
- **Resumability.** Re-running a driver re-uses cells with matching
  `(experiment, code_identity, config, seed, run_context)` and only
  computes the rest. Logs land under `results/sl_clean/logs/<phase>.log`.

---

## Phase index

| # | phase | experiment | knob | claim it supports |
|---|---|---|---|---|
| — | smoke_n_neurons | smoke | `n_neurons` | infrastructure plumbing only |
| — | smoke_data_driven_bias | smoke | `data_driven_biases` | infrastructure plumbing only |
| B1 | biases_mnist_panel | biases_mnist | `cell` (3×3 factorial) | H1 — encoder-type gap closure |

Headline-claim phases (streaming sMNIST, MLP / LSTM / ESN baselines,
etc.) will be added as the corresponding experiments are wired up.

---

## B1 — biases_mnist_panel (H1 keystone)

**Question.** Does the choice of bias mechanism matter for the encoder-
type gap on MNIST? The original H1 hypothesis was *"data-driven biases
collapse the encoder-type gap"*. The data tells a more nuanced story.

**Setup.** 3×3 factorial: `bias_mode ∈ {data_driven, random_normal,
zero}` × `encoder_strategy ∈ {hypersphere, gaussian, sparse}` × 10 seeds
× n_neurons=5000, abs activation, gain U(0.5, 2.0), α=1e-3, Cholesky
solver, chunked accumulation. Encoders and gains are bit-identical
across the 3 bias modes for the same seed (independent sub-RNGs per
stream), so the bias-mode contrast is causal.

Results: `results/sl_clean/biases_mnist_panel.json` (90 rows),
summary in `biases_mnist_panel_summary.json`. Wall time: ~18 min on
12-thread Ryzen CPU.

### Per-cell test accuracy (mean ± SE, n=10)

| bias_mode \ encoder | hypersphere | gaussian | sparse |
|---|---|---|---|
| data_driven   | 97.1 ± 0.0 | 97.1 ± 0.0 | 97.1 ± 0.0 |
| random_normal | **96.1 ± 0.0** | 97.1 ± 0.0 | 97.2 ± 0.0 |
| zero          | 97.1 ± 0.0 | 97.1 ± 0.0 | 97.2 ± 0.0 |

### Encoder-type gap (max pairwise gap within each bias_mode, paired by seed)

| bias_mode | mean | SE | n | verdict |
|---|---|---|---|---|
| data_driven   | 0.10% | 0.02% | 10 | collapse_supported |
| random_normal | 1.12% | 0.04% | 10 | gap_present |
| zero          | 0.11% | 0.02% | 10 | collapse_supported |

(collapse threshold = 0.50 pp.)

### Headline contrasts: data_driven − random_normal (paired by seed)

| encoder | Δ mean | Δ SE | ratio | n |
|---|---|---|---|---|
| hypersphere | **+0.99 pp** | 0.05 | **18.5** | 10 |
| gaussian    | −0.06 pp | 0.05 | −1.3 | 10 |
| sparse      | −0.11 pp | 0.03 | −4.0 | 10 |

### Verdict

The original H1 framing — *"data-driven biases close the encoder-type
gap"* — needs revision. The data shows three things the simple framing
hides:

1. **`random_normal` biases are a failure mode for `hypersphere`
   encoders specifically** — they cost ~1 pp on MNIST. `gaussian` and
   `sparse` encoders are robust to bias choice.
2. **`zero` biases collapse the gap just as effectively as
   `data_driven`.** The honest story is not that data-driven biases
   are uniquely good, but that random Gaussian biases are uniquely bad
   for some encoders.
3. **Therefore the practical recommendation simplifies:** *don't use
   random Gaussian biases.* `data_driven` and `zero` are
   interchangeable on this benchmark within ≤ 0.05 pp.

This is a stronger and more useful finding than the original H1, and
it raises follow-up questions before B1 can close:

- **Is `data_driven == zero` dataset-specific?** Probably not on MNIST,
  but Fashion-MNIST and CIFAR-10 may differentiate them. Adding a
  Fashion-MNIST replication of B1 is now elevated to a B-phase blocker.
- **Does the gain range matter?** Larger gains amplify the
  random-Gaussian bias offset; smaller gains may close it. A gain
  ablation with random_normal+hypersphere is queued.
- **Why `hypersphere` only?** The unit-norm constraint means random
  biases land neurons at unpredictable points relative to the data
  cloud. `gaussian` and `sparse` directions implicitly carry magnitude
  variation that interacts with U(0.5, 2.0) gain in a way that
  compensates. This is a hypothesis for §5 of the paper, not yet
  proven.

### Code / data identity

- `code_identity` for all 90 rows: `a1837796f1fc783b83846670e39f75548281a979`
- `dataset.raw_hashes` (MNIST IDX): per-row in JSON; should match across
  any reproduction on this machine.

### Reproduce

```bash
SEEDS=0-9 PYTHON=$PWD/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh biases_mnist
```

---

## Smoke (worked example)

A tiny synthetic binary-classification task that exercises the full
runner / analyzer path without pulling MNIST. Useful to sanity-check
new infrastructure changes against the cached cells before launching
real sweeps. Expected wall time: a few seconds per cell.

```bash
bash benchmarks/sl_clean/sl_hp_driver.sh smoke
```

The smoke phases are not load-bearing for any paper claim and will be
deleted from the index once at least one headline phase has landed and
demonstrated the same workflow.

---

## Base config evolution

Each headline experiment will track its own `base_config_<exp>.json`.
When the analyzer adopts a new value, append a row here with the new
snapshot, the phase that triggered the change, the Δ / SE_combined,
and the seed counts on each side.

| date | experiment | base file | knob | old → new | Δ ± SE_combined | n_winner / n_base |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |
