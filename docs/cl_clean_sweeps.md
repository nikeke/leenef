# cl_clean phase log — Paper II (Continual Learning)

This document is the chronological log of `cl_clean` phases.  It
mirrors `docs/sl_clean_sweeps.md`: one section per phase, each
recording configs, sentinels, dates, and raw verdicts.  Numbers
quoted here are the **as-run** values; the manuscript may round them.

Phase letters match the plan in
`~/.copilot/session-state/.../plan.md` §6 but **never appear in the
manuscript** — Paper II refers to phases by descriptive name.

## Phase A — Scaffolding

Date: 2026-05-04..

* Branch: `paper-ii/cl-scaffold`.
* Created `benchmarks/cl_clean/{datasets.py, experiments.py,
  cl_repro.py, cl_phase_runner.py, analyze_cl_sweep.py,
  analyze_cl_panel.py, cl_hp_driver.sh, baselines/}`.
* Registered experiments: `smoke`,
  `nef_continual_split_mnist`, `nef_continual_permuted_mnist`,
  `nef_continual_split_cifar10`, `nef_continual_split_cifar100`.
* Cells per panel: `nef_accumulate`, `nef_joint`, `nef_reset`,
  `mlp_finetune`, `mlp_joint`, `mlp_ewc`, `mlp_er`.
* Tests in `tests/test_cl_clean.py` cover code-identity format,
  runner idempotency, deterministic task streams, and the smoke
  run.

## Phase B — headline panels (delegated worktree)

(Pending kickoff in `paper-ii/cl-experiments` worktree.)

### B1 — Joint-equivalence panel (H1)

Sentinel: `results/cl_clean/<panel>_done` for each of the four
datasets.  Verdict gate: `max_abs_gap` between `nef_accumulate` and
`nef_joint` ≤ 0.05 pp on every panel.

### B2 — Cross-method panel (H5)

Same panel; the MLP cells produce the H5 comparison.

### B3 — Capacity matrix (H2)

Pending; design under §6.B3 of the plan.

### B4 — Center-adaptation falsifier (H3)

Pending; design under §6.B4 of the plan.

### B5 — Solver-mode equivalence + α regime (H4)

Pending; design under §6.B5 of the plan.

## Phase C — Visual extension (Colab GPU)

(Blocked on Paper I v2 ConvNEF chapter landing.)
