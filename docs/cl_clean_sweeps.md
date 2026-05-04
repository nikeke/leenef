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
* Registered B3 capacity experiments:
  `nef_capacity_permuted_mnist_t{5,10,20,50,100}` and
  `nef_capacity_permuted_cifar10_t{5,10,20}`.
* Cells per panel: `nef_accumulate`, `nef_joint`, `nef_reset`,
   `mlp_finetune`, `mlp_joint`, `mlp_ewc`, `mlp_er`.
* Additional B3-B5 driver targets landed after the review-fix pass:
  `b3_capacity`, `b4_centers`, and `b5_solvers`.  These are a new
  code identity relative to any already-launched B1/B2 panels; do not
  mix old-code and new-code rows in one aggregate.
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

Runnable via:

```bash
PYTHON=$PWD/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b3_capacity
```

Defaults: Permuted-MNIST `n_neurons ∈ {500,1000,2000,5000,10000}`,
`n_tasks ∈ {5,10,20,50,100}`, seeds 0-4; Permuted-CIFAR-10
confirmation `n_neurons ∈ {1000,2000,5000,10000}`,
`n_tasks ∈ {5,10,20}`, seeds 0-2; four Permuted-MNIST
joint-equivalence spot checks at the grid extremes.  Override with
`B3_MNIST_N`, `B3_CIFAR_N`, `B3_MNIST_TASKS`, `B3_CIFAR_TASKS`,
`B3_MNIST_SEEDS`, and `B3_CIFAR_SEEDS`.  The base config carries
`"device": "cpu"`; switch it to `"cuda"` for a GPU execution branch
and keep CPU/GPU timing rows labeled by their config.

### B4 — Center-adaptation falsifier (H3)

Runnable via:

```bash
PYTHON=$PWD/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b4_centers
```

Runs Split-MNIST and Split-CIFAR-10 over
`centers_mode ∈ {none,first_task,all_tasks,growing}` with seeds 0-4
by default.  The `growing` condition is the checkpoint-era negative
control: neurons are partitioned by task, each new group receives
centers from the current task, future groups are inactive for earlier
tasks, and no replay is used.

### B5 — Solver-mode equivalence + α regime (H4)

Runnable via:

```bash
PYTHON=$PWD/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b5_solvers
```

Runs Split-MNIST, Permuted-MNIST, and Split-CIFAR-10 over
`solver_mode ∈ {accumulate,woodbury_batch,woodbury_online}` and
`α ∈ {10^-3,10^-2,10^-1,1,10,10^2,10^3}` with seeds 0-4 by default.
The online Woodbury row records refresh-drift diagnostics
(`refresh_decoder_max_abs_diff`, refreshed accuracy) in addition to
the online-decoder accuracy.  The base config carries
`"device": "cpu"`; switch it to `"cuda"` for GPU execution when
appropriate.

## Phase C — Visual extension (Colab GPU)

(Blocked on Paper I v2 ConvNEF chapter landing.)
