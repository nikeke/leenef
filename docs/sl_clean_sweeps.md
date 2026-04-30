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

Headline-claim phases (data-driven biases, streaming sMNIST, MLP / LSTM
/ ESN baselines, etc.) will be added as the corresponding experiments
are wired up.

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
