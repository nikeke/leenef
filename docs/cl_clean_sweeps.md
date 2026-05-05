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

#### 2026-05-04 — Split-MNIST (`nef_continual_split_mnist`)

* Code identity: `ed891a3d7bbefb532fe876847b635a1fb871aa43`.
* Command family: `PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python SEEDS=0-9 bash benchmarks/cl_clean/cl_hp_driver.sh nef_continual_split_mnist`.
* Thread caps at launch: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
  `OPENBLAS_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`.
* Base config: `results/cl_clean/base_config_nef_continual_split_mnist.json`
  (`n_neurons=5000`, `alpha=1e-3`, `centers_mode=first_task`,
  `fit_chunk_size=4096`).
* Cells: `nef_accumulate`, `nef_joint`, `nef_reset`, `mlp_finetune`,
  `mlp_joint`, `mlp_ewc`, `mlp_er`; seeds 0-9; 70 rows complete.
* Sentinel: `results/cl_clean/nef_continual_split_mnist_panel_done`.

Per-cell summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| Cell | n | Accuracy | Fit time |
| --- | ---: | ---: | ---: |
| `nef_accumulate` | 10 | 97.022% ± 0.034% | 31.13s ± 0.27s |
| `nef_joint` | 10 | 97.022% ± 0.034% | 31.87s ± 0.21s |
| `nef_reset` | 10 | 19.811% ± 0.224% | 7.07s ± 0.10s |
| `mlp_finetune` | 10 | 19.737% ± 0.229% | 9.22s ± 0.18s |
| `mlp_joint` | 10 | 98.056% ± 0.048% | 54.40s ± 1.71s |
| `mlp_ewc` | 10 | 19.979% ± 0.266% | 20.33s ± 0.21s |
| `mlp_er` | 10 | 86.533% ± 0.616% | 13.98s ± 0.17s |

Joint-equivalence summary:

* `mean_signed_gap(accumulate - joint) = +0.0000 pp`
* `max_abs_gap(accumulate - joint) = 0.0000 pp`
* `n = 10`
* Verdict: **joint-equivalence supported (≤ 0.05 pp)**.

Cross-method / forgetting summary:

* Forgetting-gap proxy (`accumulate - reset`): `+77.211 ± 0.213 pp`.
* `mlp_finetune` and `mlp_ewc` both collapse to ~20% final accuracy on
  the class-incremental split; replay rescues performance to 86.5%, but
  still trails `nef_accumulate` by 10.5 pp.
* `nef_accumulate` trails `mlp_joint` by 1.034 pp while using 1.75× less
  end-to-end wall-clock fitting time (`31.13s` vs `54.40s`).
* Adoption verdicts: **H1 supported strongly** on Split-MNIST; **H5
  supported against finetune/EWC**, with replay competitive but clearly
  below the NEF accumulate/joint pair.

### B2 — Cross-method panel (H5)

Same panel; the MLP cells produce the H5 comparison.

#### 2026-05-04 — Split-CIFAR-10 (`nef_continual_split_cifar10`)

* Code identity: `ed891a3d7bbefb532fe876847b635a1fb871aa43`.
* Command family: `PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python SEEDS=0-9 bash benchmarks/cl_clean/cl_hp_driver.sh nef_continual_split_cifar10`.
* Thread caps at launch: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
  `OPENBLAS_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`.
* Base config: `results/cl_clean/base_config_nef_continual_split_cifar10.json`
  (`n_neurons=5000`, `alpha=1e-3`, `centers_mode=first_task`,
  `fit_chunk_size=4096`).
* Cells: `nef_accumulate`, `nef_joint`, `nef_reset`, `mlp_finetune`,
  `mlp_joint`, `mlp_ewc`, `mlp_er`; seeds 0-9; 70 rows complete.
* Sentinel: `results/cl_clean/nef_continual_split_cifar10_panel_done`.

Per-cell summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| Cell | n | Accuracy | Fit time |
| --- | ---: | ---: | ---: |
| `nef_accumulate` | 10 | 49.484% ± 0.081% | 34.84s ± 0.23s |
| `nef_joint` | 10 | 49.484% ± 0.081% | 35.43s ± 0.26s |
| `nef_reset` | 10 | 16.122% ± 0.455% | 7.81s ± 0.21s |
| `mlp_finetune` | 10 | 12.862% ± 0.912% | 17.06s ± 0.13s |
| `mlp_joint` | 10 | 51.863% ± 0.158% | 87.78s ± 1.96s |
| `mlp_ewc` | 10 | 16.123% ± 0.477% | 37.46s ± 0.20s |
| `mlp_er` | 10 | 12.527% ± 1.075% | 24.54s ± 0.21s |

Joint-equivalence summary:

* `mean_signed_gap(accumulate - joint) = +0.0000 pp`
* `max_abs_gap(accumulate - joint) = 0.0000 pp`
* `n = 10`
* Verdict: **joint-equivalence supported (≤ 0.05 pp)**.

Cross-method / forgetting summary:

* Forgetting-gap proxy (`accumulate - reset`): `+33.362 ± 0.497 pp`.
* The SGD baselines remain far below the NEF pair on this
  class-incremental visual split: `mlp_finetune` and `mlp_er` stay near
  12-13%, and `mlp_ewc` reaches only 16.1%.
* `mlp_joint` is the accuracy leader at 51.863%, but the gap over
  `nef_accumulate` is only 2.379 pp while taking 2.52× more end-to-end
  wall-clock fitting time (`87.78s` vs `34.84s`).
* Adoption verdicts: **H1 supported strongly** on Split-CIFAR-10; **H5
  supported against the continual SGD baselines**, with MLP-joint still
  the best absolute-accuracy upper bound on this panel.

#### 2026-05-04 — Split-CIFAR-100 (`nef_continual_split_cifar100`)

* Code identity: `ed891a3d7bbefb532fe876847b635a1fb871aa43`.
* Command family: `PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python SEEDS=0-9 bash benchmarks/cl_clean/cl_hp_driver.sh nef_continual_split_cifar100`.
* Thread caps at launch: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
  `OPENBLAS_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`.
* Base config: `results/cl_clean/base_config_nef_continual_split_cifar100.json`
  (`n_neurons=5000`, `alpha=1e-3`, `centers_mode=first_task`,
  `fit_chunk_size=4096`).
* Cells: `nef_accumulate`, `nef_joint`, `nef_reset`, `mlp_finetune`,
  `mlp_joint`, `mlp_ewc`, `mlp_er`; seeds 0-9; 70 rows complete.
* Sentinel: `results/cl_clean/nef_continual_split_cifar100_panel_done`.

Per-cell summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| Cell | n | Accuracy | Fit time |
| --- | ---: | ---: | ---: |
| `nef_accumulate` | 10 | 24.147% ± 0.080% | 35.77s ± 0.34s |
| `nef_joint` | 10 | 24.147% ± 0.080% | 36.52s ± 0.29s |
| `nef_reset` | 10 | 3.943% ± 0.146% | 4.66s ± 0.08s |
| `mlp_finetune` | 10 | 1.080% ± 0.046% | 17.97s ± 0.38s |
| `mlp_joint` | 10 | 23.809% ± 0.132% | 180.26s ± 10.00s |
| `mlp_ewc` | 10 | 4.115% ± 0.224% | 40.41s ± 0.12s |
| `mlp_er` | 10 | 2.035% ± 0.363% | 17.70s ± 0.10s |

Joint-equivalence summary:

* `mean_signed_gap(accumulate - joint) = +0.0000 pp`
* `max_abs_gap(accumulate - joint) = 0.0000 pp`
* `n = 10`
* Verdict: **joint-equivalence supported (≤ 0.05 pp)**.

Cross-method / forgetting summary:

* Forgetting-gap proxy (`accumulate - reset`): `+20.204 ± 0.128 pp`.
* All continual SGD baselines collapse to the 1-4% range on this
  100-class class-incremental split.
* The NEF pair is the best-performing family on the panel:
  `nef_accumulate` exceeds `mlp_joint` by 0.338 pp while using 5.04×
  less end-to-end wall-clock fitting time (`35.77s` vs `180.26s`).
* Adoption verdicts: **H1 supported strongly** on Split-CIFAR-100; **H5
  supported strongly**, including against the joint MLP upper bound on
  this panel.

#### 2026-05-05 — Permuted-MNIST (`nef_continual_permuted_mnist`)

* Code identity: `ed891a3d7bbefb532fe876847b635a1fb871aa43`.
* Command family: `PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python SEEDS=0-9 bash benchmarks/cl_clean/cl_hp_driver.sh nef_continual_permuted_mnist`.
* Thread caps at launch: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
  `OPENBLAS_NUM_THREADS=4`, `NUMEXPR_NUM_THREADS=4`.
* Base config: `results/cl_clean/base_config_nef_continual_permuted_mnist.json`
  (`n_neurons=5000`, `alpha=1e-3`, `centers_mode=first_task`,
  `fit_chunk_size=4096`).
* Cells: `nef_accumulate`, `nef_joint`, `nef_reset`, `mlp_finetune`,
  `mlp_joint`, `mlp_ewc`, `mlp_er`; seeds 0-9; 70 rows complete.
* Sentinel: `results/cl_clean/nef_continual_permuted_mnist_panel_done`.

Per-cell summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| Cell | n | Accuracy | Fit time |
| --- | ---: | ---: | ---: |
| `nef_accumulate` | 10 | 94.498% ± 0.021% | 154.06s ± 5.13s |
| `nef_joint` | 10 | 94.498% ± 0.021% | 90.79s ± 4.93s |
| `nef_reset` | 10 | 27.606% ± 0.475% | 16.33s ± 0.02s |
| `mlp_finetune` | 10 | 71.072% ± 0.662% | 20.57s ± 0.02s |
| `mlp_joint` | 10 | 97.975% ± 0.024% | 91.76s ± 0.13s |
| `mlp_ewc` | 10 | 95.599% ± 0.194% | 37.50s ± 0.03s |
| `mlp_er` | 10 | 93.875% ± 0.145% | 26.12s ± 0.03s |

Joint-equivalence summary:

* `mean_signed_gap(accumulate - joint) = +0.0000 pp`
* `max_abs_gap(accumulate - joint) = 0.0000 pp`
* `n = 10`
* Verdict: **joint-equivalence supported (≤ 0.05 pp)**.

Cross-method / forgetting summary:

* Forgetting-gap proxy (`accumulate - reset`): `+66.892 ± 0.465 pp`.
* The fixed-encoder NEF story remains exact but no longer dominates the
  stronger gradient baselines on this permutation-shift regime:
  `mlp_ewc` exceeds `nef_accumulate` by 1.101 pp and `mlp_joint` leads by
  3.477 pp.
* `nef_accumulate` still stays ahead of replay by 0.623 pp and far ahead
  of naive finetuning, but continual accumulation is also slower than the
  one-shot joint solve here (`154.06s` vs `90.79s`).
* Adoption verdicts: **H1 supported strongly** on Permuted-MNIST; **H5
  mixed / not supported as a blanket claim** on this panel, because EWC
  and the joint MLP both outperform the NEF pair in final accuracy.

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
