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
   `nef_continual_split_cifar10`, `nef_continual_split_cifar100`,
   `c1_convnef_split_cifar10`.
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
PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python \
    bash benchmarks/cl_clean/cl_hp_driver.sh b3_capacity
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

#### 2026-05-05 to 2026-05-08 — Permuted-MNIST capacity grid (`nef_capacity_permuted_mnist_t{5,10,20,50,100}`)

* Code identity: `ed891a3d7bbefb532fe876847b635a1fb871aa43`.
* Command family: `PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b3_capacity`.
* Base configs:
  `results/cl_clean/base_config_b3_capacity.json`,
  `results/cl_clean/base_config_b3_capacity_spot_n500.json`, and
  `results/cl_clean/base_config_b3_capacity_spot_n10000.json`
  (`alpha=1e-3`, `centers_mode=first_task`, `fit_chunk_size=4096`,
  `device=cpu`).
* Completed artifacts:
  `b3_capacity_permuted_mnist_t{5,10,20,50,100}.json`,
  `b3_capacity_spot_permuted_mnist_t{5,100}_n{500,10000}.json`, and
  their paired logs.
* Row counts: 125 grid rows (5 task counts × 5 neuron counts × 5 seeds)
  plus 40 spot-check rows (4 panels × 2 cells × 5 seeds).
* Operational note: the final `t100,n=10000` spot-check was resumed as
  temporary `*_accum_resume` and `*_joint_resume` phases to keep CPU
  usage near the target range; those outputs were merged back into the
  canonical `b3_capacity_spot_permuted_mnist_t100_n10000.{json,log}`
  artifact after completion.

Accuracy matrix (`test_accuracy`, % mean ± SE):

| Tasks \ Neurons | 500 | 1000 | 2000 | 5000 | 10000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 82.6 ± 0.1 | 88.0 ± 0.1 | 91.6 ± 0.0 | 94.5 ± 0.0 | 95.9 ± 0.0 |
| 10 | 73.9 ± 0.1 | 82.5 ± 0.0 | 88.0 ± 0.0 | 92.4 ± 0.0 | 94.5 ± 0.0 |
| 20 | 61.9 ± 0.1 | 73.8 ± 0.1 | 82.2 ± 0.0 | 89.1 ± 0.0 | 92.4 ± 0.0 |
| 50 | 44.1 ± 0.1 | 56.9 ± 0.2 | 69.4 ± 0.1 | 81.4 ± 0.0 | 87.3 ± 0.0 |
| 100 | 34.2 ± 0.2 | 43.9 ± 0.1 | 56.3 ± 0.2 | 71.8 ± 0.1 | 80.8 ± 0.0 |

Joint-equivalence spot checks (`test_accuracy`, % mean ± SE):

| Spot check | `nef_accumulate` | `nef_joint` | Mean gap (acc - joint) | Max abs gap |
| --- | ---: | ---: | ---: | ---: |
| `t5, n=500` | 82.552% ± 0.112% | 82.552% ± 0.112% | +0.0000 pp | 0.0000 pp |
| `t5, n=10000` | 95.910% ± 0.026% | 95.910% ± 0.026% | +0.0000 pp | 0.0000 pp |
| `t100, n=500` | 34.170% ± 0.189% | 34.170% ± 0.189% | +0.0000 pp | 0.0000 pp |
| `t100, n=10000` | 80.783% ± 0.032% | 80.783% ± 0.032% | +0.0000 pp | 0.0000 pp |

Capacity / stability summary:

* The approximate 80%-accuracy frontier lies almost exactly on
  `(n_neurons, n_tasks) = (500,5), (1000,10), (2000,20), (5000,50), (10000,100)`,
  i.e. about `100 neurons/task` or
  $n / (T \cdot 10) \approx 10$ for the 10-way output head.
* Per-task accuracy dispersion at 100 tasks falls from `3.1 ± 0.1%` at
  500 neurons to `0.6 ± 0.0%` at 10000 neurons, so higher neuron budgets
  buy both mean accuracy and cross-task uniformity.
* CPU fit time on the same 100-task frontier scales from
  `65.1 ± 0.3s` (500 neurons) to `6727.5 ± 57.2s` (10000 neurons),
  consistent with the expected roughly linear-in-tasks and strongly
  superlinear-in-neurons cost profile.
* Verdict: **H2 supported on Permuted-MNIST**.  The additivity-backed
  joint-equivalence claim remains exact at both low/high-capacity and
  low/high-task extremes, while the usable accuracy frontier follows the
  predicted near-linear neuron-per-task rule.

#### 2026-05-05 to 2026-05-06 — Permuted-CIFAR-10 confirmation (`nef_capacity_permuted_cifar10_t{5,10,20}`)

* Code identity: `ed891a3d7bbefb532fe876847b635a1fb871aa43`.
* Command family: `PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b3_capacity`.
* Base config: `results/cl_clean/base_config_b3_capacity.json`
  (`alpha=1e-3`, `centers_mode=first_task`, `fit_chunk_size=4096`,
  `device=cpu`).
* Completed artifacts:
  `b3_capacity_permuted_cifar10_t{5,10,20}.json` and paired logs.
* Row counts: 36 rows (3 task counts × 4 neuron counts × 3 seeds).

Accuracy matrix (`test_accuracy`, % mean ± SE):

| Tasks \ Neurons | 1000 | 2000 | 5000 | 10000 |
| --- | ---: | ---: | ---: | ---: |
| 5 | 39.5 ± 0.1 | 42.5 ± 0.1 | 45.8 ± 0.1 | 48.0 ± 0.2 |
| 10 | 36.1 ± 0.1 | 39.4 ± 0.0 | 43.5 ± 0.0 | 45.8 ± 0.0 |
| 20 | 32.6 ± 0.1 | 36.1 ± 0.1 | 40.3 ± 0.0 | 43.0 ± 0.0 |

Confirmation summary:

* The same monotone scaling pattern holds — more neurons buy accuracy,
  and more tasks cost accuracy — but the absolute frontier is far lower
  than on Permuted-MNIST.
* Even at `10000` neurons, the 20-task panel reaches only
  `43.0 ± 0.0%`, so the `100 neurons/task` frontier constant is **not**
  a cross-dataset universal.
* Fit time at 20 tasks rises from `51.3 ± 0.1s` (1000 neurons) to
  `1269.7 ± 54.0s` (10000 neurons), with per-task std staying tightly
  bounded around `0.3-0.4%`.
* Verdict: **B3 cross-dataset confirmation is qualitative rather than
  law-matching**.  CIFAR-10 preserves the capacity trend but suggests
  the constant in the scaling law is dataset-dependent.

### B4 — Center-adaptation falsifier (H3)

Runnable via:

```bash
PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python \
    bash benchmarks/cl_clean/cl_hp_driver.sh b4_centers
```

Runs Split-MNIST and Split-CIFAR-10 over
`centers_mode ∈ {none,first_task,all_tasks,growing}` with seeds 0-4
by default.  The `growing` condition is the checkpoint-era negative
control: neurons are partitioned by task, each new group receives
centers from the current task, future groups are inactive for earlier
tasks, and no replay is used.

#### 2026-05-05 — Center-adaptation falsifier (`b4_centers`)

* Code identity: `ed891a3d7bbefb532fe876847b635a1fb871aa43`.
* Command family: `PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b4_centers`.
* Thread caps at launch: `OMP_NUM_THREADS=5`, `MKL_NUM_THREADS=5`,
  `OPENBLAS_NUM_THREADS=5`, `NUMEXPR_NUM_THREADS=5`.
* Base config: `results/cl_clean/base_config_b4_centers.json`
  (`n_neurons=5000`, `alpha=1e-3`, `centers_mode=first_task`,
  `fit_chunk_size=4096`, `device=cpu`).
* Panels: `b4_centers_split_mnist` and `b4_centers_split_cifar10`;
  `centers_mode ∈ {none,first_task,all_tasks,growing}`; seeds 0-4;
  20 rows complete in each JSON.
* Sentinel: `results/cl_clean/b4_centers_done`.

Split-MNIST summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| centers_mode | n | Accuracy | Fit time |
| --- | ---: | ---: | ---: |
| `all_tasks` | 5 | 97.018% ± 0.017% | 26.23s ± 0.07s |
| `first_task` | 5 | 96.980% ± 0.053% | 26.12s ± 0.08s |
| `none` | 5 | 96.008% ± 0.071% | 25.83s ± 0.16s |
| `growing` | 5 | 19.772% ± 0.188% | 24.27s ± 0.26s |

Split-MNIST adoption check:

* Winner by mean: `all_tasks` at 97.018%.
* Current base: `first_task` at 96.980%.
* `Δ = 0.038 pp`, `SE_combined = 0.055 pp`, `ratio = 0.69`.
* Verdict: **keep `first_task`**; the apparent `all_tasks` gain stays
  within 1 combined standard error.

Split-CIFAR-10 summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| centers_mode | n | Accuracy | Fit time |
| --- | ---: | ---: | ---: |
| `first_task` | 5 | 49.640% ± 0.103% | 29.82s ± 0.11s |
| `all_tasks` | 5 | 49.524% ± 0.259% | 29.98s ± 0.07s |
| `none` | 5 | 48.092% ± 0.129% | 29.82s ± 0.14s |
| `growing` | 5 | 16.434% ± 0.700% | 28.06s ± 0.24s |

Split-CIFAR-10 adoption check:

* Winner by mean: `first_task` at 49.640%.
* Current base: `first_task` at 49.640%.
* Verdict: **base already optimal** on this panel.

Phase verdict:

* `none` costs about 1.0 pp on Split-MNIST and 1.5 pp on Split-CIFAR-10,
  so some data-driven centering is clearly useful.
* `all_tasks` does not beat `first_task` decisively on either panel.
* The `growing` negative control collapses on both datasets, ruling out
  the checkpoint-era idea that continually adding task-local center pools
  improves retention.
* Adoption verdict: **H3 supported** in the conservative form — fixed
  `first_task` centers remain the right default, and continual
  recentering shows no compelling benefit.

### B5 — Solver-mode equivalence + α regime (H4)

Runnable via:

```bash
PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python \
    bash benchmarks/cl_clean/cl_hp_driver.sh b5_solvers
```

Runs Split-MNIST, Permuted-MNIST, and Split-CIFAR-10 over
`solver_mode ∈ {accumulate,woodbury_batch,woodbury_online}` and
`α ∈ {10^-3,10^-2,10^-1,1,10,10^2,10^3}` with seeds 0-4 by default.
The online Woodbury row records refresh-drift diagnostics
(`refresh_decoder_max_abs_diff`, refreshed accuracy) in addition to
the online-decoder accuracy.  The base config carries
`"device": "cpu"`; switch it to `"cuda"` for GPU execution when
appropriate.

#### 2026-05-07 — Solver-mode equivalence + α regime (`b5_solvers`)

* Code identity: `ed891a3d7bbefb532fe876847b635a1fb871aa43`.
* Command family: `PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b5_solvers`.
* Thread caps at launch: `OMP_NUM_THREADS=5`, `MKL_NUM_THREADS=5`,
  `OPENBLAS_NUM_THREADS=5`, `NUMEXPR_NUM_THREADS=5`.
* Base config: `results/cl_clean/base_config_b5_solvers.json`
  (`n_neurons=5000`, `cell=nef_accumulate`,
  `solver_alpha_cell=accumulate_a0.001`, `fit_chunk_size=4096`,
  `woodbury_chunk_size=100`, `device=cpu`).
* Panels: `b5_solvers_split_mnist`, `b5_solvers_permuted_mnist`, and
  `b5_solvers_split_cifar10`; 105 rows complete in each JSON.
* Sentinel: `results/cl_clean/b5_solvers_done`.

Split-MNIST mode summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| Solver family | Best cell | n | Accuracy | Fit time |
| --- | --- | ---: | ---: | ---: |
| `accumulate` | `accumulate_a0.001` | 5 | 96.980% ± 0.053% | 26.93s ± 0.09s |
| `woodbury_batch` | `woodbury_batch_a0.001` | 5 | 96.996% ± 0.047% | 164.91s ± 0.44s |
| `woodbury_online` | `woodbury_online_a0.001` | 5 | 96.996% ± 0.047% | 239.80s ± 0.56s |

Split-MNIST adoption check:

* Winner by mean: `woodbury_batch_a0.001` at 96.996%.
* Current base: `accumulate_a0.001` at 96.980%.
* `Δ = 0.02 pp`, `SE_combined = 0.07 pp`, `ratio = 0.23`.
* Verdict: **keep `accumulate_a0.001`**; the Woodbury edge stays well
  within 1 combined standard error.

Permuted-MNIST mode summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| Solver family | Best cell | n | Accuracy | Fit time |
| --- | --- | ---: | ---: | ---: |
| `accumulate` | `accumulate_a0.001` | 5 | 94.518% ± 0.029% | 124.81s ± 1.54s |
| `woodbury_batch` | `woodbury_batch_a0.01` | 5 | 94.557% ± 0.036% | 881.06s ± 14.55s |
| `woodbury_online` | `woodbury_online_a0.01` | 5 | 94.557% ± 0.036% | 1135.10s ± 0.91s |

Permuted-MNIST adoption check:

* Winner by mean: `woodbury_batch_a0.01` at 94.557%.
* Current base: `accumulate_a0.001` at 94.518%.
* `Δ = 0.04 pp`, `SE_combined = 0.05 pp`, `ratio = 0.83`.
* Verdict: **keep `accumulate_a0.001`**; the Woodbury gain again stays
  within 1 combined standard error.

Split-CIFAR-10 mode summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| Solver family | Best cell | n | Accuracy | Fit time |
| --- | --- | ---: | ---: | ---: |
| `accumulate` | `accumulate_a0.01` | 5 | 50.448% ± 0.054% | 29.25s ± 0.10s |
| `woodbury_batch` | `woodbury_batch_a100` | 5 | 50.432% ± 0.040% | 137.21s ± 0.88s |
| `woodbury_online` | `woodbury_online_a100` | 5 | 50.430% ± 0.040% | 202.19s ± 0.11s |

Split-CIFAR-10 adoption check:

* Winner by mean: `accumulate_a0.01` at 50.448%.
* Current base: `accumulate_a0.001` at 49.640%.
* `Δ = 0.81 pp`, `SE_combined = 0.12 pp`, `ratio = 6.94`.
* Verdict: **adopt `accumulate_a0.01`** for the Split-CIFAR-10-scale
  continual setting.

Phase verdict:

* H4 is supported in the **solver-equivalence** sense: once `α` is
  retuned, `accumulate`, `woodbury_batch`, and `woodbury_online` land in
  nearly the same accuracy band on all three panels.
* The solver family itself is not the useful knob.  At matched
  accuracy, `accumulate` remains about 4.7-7.1x faster in fit time than
  the best `woodbury_batch` row and about 6.9-9.1x faster than the best
  `woodbury_online` row.
* The online-refresh diagnostics stay numerically tight
  (`refresh_decoder_max_abs_diff <= 2.3e-05`,
  mean refreshed-accuracy gap `<= 0.002 pp`), so the online Woodbury
  path is not failing because of drift; it is simply slower.
* Adoption verdict: **keep `accumulate` as the default solver family**.
  Keep `α = 10^{-3}` for the MNIST-like panels, but use `α = 10^{-2}`
  for the Split-CIFAR-10 continual regime.

## Phase C — Visual extension (Colab GPU)

Phase C is launched via:

```bash
python -u colab/cl_clean/continual_cifar10_gpu_suite.py --seeds 0-9
```

The package lives under `colab/cl_clean/`, clones
`paper-ii/cl-scaffold` rather than `paper-ii/cl-experiments`, writes
GPU config snapshots into `results/cl_clean/colab_gpu_configs/`, and
bundles the returned JSON/log files for import back into this branch.

### C1 — ConvNEF-CL on Split-CIFAR-10

Experiment: `c1_convnef_split_cifar10`.

Cells:

* `all_data_n10000`, `first_task_n10000`, `joint_n10000`
* `all_data_n20000`, `first_task_n20000`, `joint_n20000`

Base config: `results/cl_clean/base_config_c1_convnef_split_cifar10.json`
with a Colab-generated GPU copy under `results/cl_clean/colab_gpu_configs/`.

### C2 — Flat-NEF Split-CIFAR-10 floor

Experiment: `nef_continual_split_cifar10`.

Cell:

* `nef_accumulate`

Base config: `results/cl_clean/base_config_nef_continual_split_cifar10.json`
with a Colab-generated GPU copy under `results/cl_clean/colab_gpu_configs/`.

### Imported Phase-C results

Imported from the separate Phase-C Colab bundle run against
`paper-ii/cl-scaffold`, then copied back into this branch as JSON/log
artifacts.  These rows therefore carry the scaffold-side code identity
`a89221e4b319f454fa4318318ecf3cf8eb3b1ecb`, not the CPU-phase identity
used by the B-series runs above.

#### 2026-05-05 — GPU visual-extension bundle (`gpu_c1_*`, `gpu_c2_*`)

* Source workflow: `colab/cl_clean/continual_cifar10_gpu_suite.py` on
  `paper-ii/cl-scaffold`, with artifacts manually imported into this
  worktree.
* Hardware in the imported rows: **NVIDIA L4**, CUDA enabled, PyTorch
  `2.10.0+cu128`.
* Code identity: `a89221e4b319f454fa4318318ecf3cf8eb3b1ecb`.
* Run context: Split-CIFAR-10, 5 tasks, seeds 0-9.
* Imported config snapshots:
  `results/cl_clean/base_config_gpu_c1_convnef_split_cifar10.json` and
  `results/cl_clean/base_config_gpu_c2_flat_nef_split_cifar10.json`.

Phase-C panel summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE;
end-to-end `wall_time_s`, mean ± SE):

| Panel cell | n | Accuracy | Fit time | Wall time |
| --- | ---: | ---: | ---: | ---: |
| `all_data_n10000` | 10 | 72.870% ± 0.101% | 14.94s ± 0.14s | 16.81s ± 0.53s |
| `first_task_n10000` | 10 | 72.655% ± 0.087% | 14.85s ± 0.00s | 16.32s ± 0.01s |
| `joint_n10000` | 10 | 72.870% ± 0.101% | 7.61s ± 0.00s | 8.14s ± 0.01s |
| `all_data_n20000` | 10 | 74.719% ± 0.089% | 83.46s ± 0.00s | 85.20s ± 0.01s |
| `first_task_n20000` | 10 | 74.506% ± 0.100% | 83.46s ± 0.00s | 85.20s ± 0.01s |
| `joint_n20000` | 10 | 74.719% ± 0.089% | 29.56s ± 0.01s | 30.14s ± 0.01s |
| `nef_accumulate` (`gpu_c2`) | 10 | 49.484% ± 0.078% | 0.80s ± 0.02s | 1.50s ± 0.36s |

Phase-C comparisons:

* **ConvNEF vs flat NEF floor.**  The imported flat-NEF GPU control
  (`gpu_c2_flat_nef_split_cifar10_panel`) lands at `49.484% ± 0.078%`,
  while ConvNEF reaches `72.870%` at 10000 neurons and `74.719%` at
  20000 neurons.  That is a **+23.386 pp** to **+25.235 pp** lift on the
  same device and same code identity.
* **Feature-source effect.**  Using all training images to fit the
  ConvNEF feature extractor beats fitting features on the first task
  alone by a small but consistent margin: `+0.215 ± 0.108 pp` at 10000
  neurons and `+0.213 ± 0.080 pp` at 20000 neurons.
* **Joint vs continual ConvNEF.**  `joint_n10000` matches
  `all_data_n10000` exactly (`max|Δ| = 0.0000 pp` over the imported
  seeds), and the same exact match holds at 20000 neurons.  The one-shot
  joint fit is substantially cheaper in end-to-end wall-clock:
  **2.06×** faster at 10000 neurons and **2.83×** faster at 20000.
* **Neuron-count effect.**  Doubling ConvNEF head width from 10000 to
  20000 neurons improves final accuracy by **+1.849 pp** for both the
  continual `all_data` path and the one-shot `joint` path.

Phase verdict:

* Phase C is the first result in Paper II where the visual extension
  clearly changes the ceiling rather than just the efficiency tradeoff.
* The imported ConvNEF panel turns the Split-CIFAR-10 story from
  "exact continual equivalence with a modest accuracy gap to stronger
  baselines" into "a substantially stronger visual model that beats the
  flat NEF floor by 23-25 pp."
* Within ConvNEF itself, the continual `all_data` variant does **not**
  beat the joint ConvNEF upper bound in accuracy; it matches it exactly
  at both neuron counts while paying the expected continual-processing
  time tax.

## H8 — Streaming continual follow-up (Colab GPU)

Experiment: `h8_streaming_permuted_smnist`.

Task stream: 10-task permuted sequential MNIST.  Each task is full
10-way MNIST under a fresh fixed pixel permutation, reshaped as a
784-step sequence with one scalar per timestep.  The streaming feature
map is fixed from first-task centers.

### 2026-05-09 — GPU streaming-sMNIST bundle

* Source workflow:
  `colab/cl_clean/streaming_permuted_smnist_gpu_suite.py --seeds 0-4`
  followed by the `--extended` cells.
* Hardware in the imported rows: CUDA enabled Colab GPU (L4-class in the
  runner log header).
* Code identity in all imported rows:
  `1436af7f8cda2a617d465b1f5a0e8db57d3de551`.
* Imported artifacts:
  `results/cl_clean/gpu_h8_streaming_permuted_smnist_panel.json`,
  `results/cl_clean/logs/gpu_h8_streaming_permuted_smnist_panel.log`,
  and
  `results/cl_clean/colab_gpu_configs/base_config_gpu_h8_streaming_permuted_smnist.json`.
* Rows: 60/60 complete, no error rows, 5 seeds for each of 12 cells.
* Import caveat: running `analyze_cl_panel` from the current `main`
  rejects these rows as stale because later CI/test commits changed files
  under `CODE_IDENTITY_PATHS`.  Manual audit confirms a homogeneous
  imported bundle with one code identity, one config family, and complete
  seed coverage; this is the same imported-identity situation as the
  Phase-C Colab bundle.

H8 panel summary (`test_accuracy`, mean ± SE; `fit_time_s`, mean ± SE):

| Cell | n | Accuracy | Fit time |
| --- | ---: | ---: | ---: |
| `accumulate_final_n4000` | 5 | 34.848% ± 5.562% | 46.54s ± 0.04s |
| `joint_n4000` | 5 | 34.848% ± 5.562% | 46.50s ± 0.00s |
| `accumulate_task_solve_n4000` | 5 | 34.848% ± 5.562% | 47.16s ± 0.00s |
| `woodbury_online_n4000` | 5 | 56.360% ± 0.203% | 76.82s ± 0.00s |
| `reset_n4000` | 5 | 15.841% ± 0.826% | 4.72s ± 0.00s |
| `accumulate_final_n8000` | 5 | 19.956% ± 1.555% | 103.21s ± 0.00s |
| `joint_n8000` | 5 | 19.956% ± 1.555% | 103.21s ± 0.00s |
| `woodbury_online_n8000` | 5 | 58.433% ± 0.204% | 211.05s ± 0.00s |
| `reset_n8000` | 5 | 12.136% ± 0.439% | 10.68s ± 0.00s |
| `accumulate_final_n16000` | 5 | 14.403% ± 1.207% | 248.84s ± 0.03s |
| `joint_n16000` | 5 | 14.403% ± 1.207% | 248.82s ± 0.01s |
| `woodbury_online_n16000` | 5 | 59.798% ± 0.253% | 629.36s ± 0.01s |

Pairwise checks:

* `accumulate_final` vs `joint`: `max|Δ| = 0.000000 pp` at 4000,
  8000, and 16000 neurons.
* `accumulate_task_solve_n4000` vs `accumulate_final_n4000`:
  `max|Δ| = 0.000000 pp`.
* `woodbury_online` vs `accumulate_final`: Woodbury is much higher, by
  +21.512 pp at 4000, +38.477 pp at 8000, and +45.395 pp at 16000.
* `reset` is a genuine lower bound: 15.8% at 4000 and 12.1% at 8000.

Phase verdict:

* The sufficient-statistic identity still holds through the streaming
  feature map: final accumulation and one-shot joint solve agree exactly.
* The useful model in this panel is not the exact final solve.
  `woodbury_online` is substantially more accurate and improves with
  neuron count, but no longer matches `accumulate_final`/`joint`.
* H8 is therefore a boundary result for Paper II.  It supports the
  algebraic identity but rejects the stronger hope that streaming-NEF
  continual learning is a free, solver-equivalent extension of the
  static CL story.  It should be framed as a promising online-Woodbury
  research line rather than folded into the main theorem.
