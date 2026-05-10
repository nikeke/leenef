# cl_clean handover — instructions for the executing Copilot

> Sister document to `docs/sl_clean_sweeps.md` and the Paper-I
> Phase-C handover used by `paper-i/phase-c-experiments`.  Anything
> in **AGENTS.md** still applies; this file only adds Paper-II
> specifics.

## Branch + worktree

* Scaffolding branch (this session): **`paper-ii/cl-scaffold`**.
* Execution branch (your branch): **`paper-ii/cl-experiments`**,
  created off `paper-ii/cl-scaffold` after the user has reviewed
  the scaffold.
* Worktree path: the current repository root,
  `/home/amnesia/Persistent/src/leenef-cl-experiments`
  (do **not** assume a separate `../leenef-paper-ii` worktree exists).

The scaffolding session **does not commit** to your branch and does
not modify `.git/worktrees/`.  If you need a code change to the
scaffolding (e.g., a runner bug), commit it on
`paper-ii/cl-experiments` and the user negotiates the merge back
into the scaffold branch.

## Allowed paths (you may write here)

```
benchmarks/cl_clean/**
results/cl_clean/**
docs/cl_clean_sweeps.md
docs/cl_clean_notes.md
colab/cl_clean/**
tests/test_cl_clean.py
```

## Forbidden paths (do **not** modify)

* `benchmarks/sl_clean/**`, `results/sl_clean/**` — Paper I.
* `docs/SSUP-I.md`, `docs/sl_clean_sweeps.md`,
  `docs/paper_insights.md`, `docs/build_ssup_pdf.sh`.
* `src/leenef/**` — any change here invalidates `code_identity`
  for every in-flight panel.  If a library bug shows up, surface
  it; do not patch silently.

## Phase order

**Current runnable status:** B1-B5 are wired in the scaffold.  The
driver targets are `nef_continual_*` for B1/B2, then `b3_capacity`,
`b4_centers`, and `b5_solvers`.  These targets change
`CODE_IDENTITY_PATHS` relative to the first B1/B2 scaffold: do not pull
or merge them into an executing worktree while old-code B1/B2 rows are
in flight unless you intend to stop and restart those panels.

1. **B1** joint-equivalence (4 panels × 10 seeds).  Sentinel per
   panel: `results/cl_clean/<experiment>_done`.
2. **B2** cross-method comparison.  Reuses the panels from B1;
   adds the MLP cells.
3. **B3** capacity matrix (Permuted-MNIST grid + Permuted-CIFAR-10
   spot panel).  Driver target: `b3_capacity`.
4. **B4** center-adaptation falsifier.  Driver target: `b4_centers`.
   Includes the checkpoint-era `growing` neuron-pool negative control.
5. **B5** solver-mode + α regime.  Driver target: `b5_solvers`.
   Includes `accumulate`, `woodbury_batch`, and `woodbury_online`
   cells over the fixed-alpha grid.
6. **C1, C2** ConvNEF-CL panels — launched from
    `colab/cl_clean/continual_cifar10_gpu_suite.py` on
    `paper-ii/cl-scaffold`, not from `paper-ii/cl-experiments`.
7. **H8 streaming-continual follow-up** — launched from
   `colab/cl_clean/streaming_permuted_smnist_gpu_suite.ipynb` (or its
   `.py` runner) on `main`.  This is not yet part of the Paper-II
   claims; it is the next benchmark-grade panel for the §6 forward look.

After each phase sentinel:

* for B1-B5, commit results JSON + log to `paper-ii/cl-experiments`;
* append a phase block to `docs/cl_clean_sweeps.md` with
  configs, dates, max-abs-gap, mean ± SE, and adoption verdicts;
* append a paragraph to `docs/cl_clean_notes.md` if the panel
  changed the story;
* continue straight to the next actionable phase or follow-up task;
  do **not** stop just because one phase sentinel landed.

## Kickoff sequence

```bash
cd /home/amnesia/Persistent/src/leenef-cl-experiments
source ../leenef/venv/bin/activate
PYTHONPATH=$PWD pytest tests/test_cl_clean.py -q  # sanity
PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python SEEDS=0-9 \
    bash benchmarks/cl_clean/cl_hp_driver.sh nef_continual_split_mnist
```

After B1/B2 are either complete or intentionally restarted on the new
identity, the follow-up targets are:

```bash
PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b3_capacity
PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b4_centers
PYTHONPATH=$PWD PYTHON=$PWD/../leenef/venv/bin/python bash benchmarks/cl_clean/cl_hp_driver.sh b5_solvers
```

Defaults: B3 uses 5 seeds for Permuted-MNIST and 3 seeds for the
Permuted-CIFAR-10 confirmation; B4/B5 use 5 seeds.  Override with
`B3_MNIST_SEEDS`, `B3_CIFAR_SEEDS`, `B4_SEEDS`, and `B5_SEEDS`.
The B3-B5 base configs include `"device": "cpu"`; for a Colab/GPU
handoff, change that config field to `"cuda"` before launch and keep
the resulting rows separate from CPU timing rows.

Phase C no longer uses `paper-ii/cl-experiments` as its source branch.
The notebook under `colab/cl_clean/` clones `paper-ii/cl-scaffold`
directly, writes GPU config snapshots under
`results/cl_clean/colab_gpu_configs/`, runs the standard
`benchmarks.cl_clean.cl_phase_runner`, and bundles the returned JSON/log
artifacts for manual import back into this branch.

## H8 streaming-continual GPU panel

Status: **superseded by the trace-scaled-alpha repair package below**.
Do not run the old no-argument commands for new evidence.  They are kept
only to document the historical fixed-$\alpha$ panel and to reproduce it
explicitly if needed for audit.

Use Colab/GPU for this panel.  CPU single-task permuted-sMNIST already
costs tens of minutes per seed at larger neuron counts, and the H8 panel
multiplies that by 10 task permutations and several solver modes.  A G4
runtime is preferred if available; L4 is the fallback.  The default
suite stays below the user's 80-CU ceiling; use `--extended` only if the
default panel finishes cleanly and budget remains.

Notebook pattern:

```text
colab/cl_clean/streaming_permuted_smnist_gpu_suite.ipynb
```

The notebook follows the Paper-I Colab pattern literally:

1. markdown suite intro + preferred GPU;
2. CUDA check cell;
3. shell-style clone/setup cell (`%cd`, `!git clone --branch main`, `!test -f ...`);
4. `!pip install -q -e '.[dev]'`;
5. run cell with quick command commented above the full command;
6. zip/download cell.

Quick validation:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_gpu_suite.py \
    --quick --seeds 0 --alpha 1e-3 --alpha-scale fixed
```

Historical fixed-$\alpha$ full panel:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_gpu_suite.py \
    --seeds 0-4 --alpha 1e-3 --alpha-scale fixed
```

Default cells:

```text
accumulate_final_n4000
accumulate_task_solve_n4000
woodbury_online_n4000
joint_n4000
reset_n4000
accumulate_final_n8000
woodbury_online_n8000
joint_n8000
reset_n8000
```

Optional extension (only after the default panel lands):

```bash
python -u colab/cl_clean/streaming_permuted_smnist_gpu_suite.py \
    --seeds 0-4 --extended --alpha 1e-3 --alpha-scale fixed
```

The extension adds:

```text
accumulate_final_n16000
joint_n16000
woodbury_online_n16000
```

Expected outputs:

```text
results/cl_clean/gpu_h8_streaming_permuted_smnist_panel.json
results/cl_clean/logs/gpu_h8_streaming_permuted_smnist_panel.log
results/cl_clean/colab_gpu_configs/base_config_gpu_h8_streaming_permuted_smnist.json
results/cl_clean/cl_clean_streaming_smnist_gpu_bundle.zip
```

Import protocol after download:

1. unzip the bundle into the repository root;
2. run `python -m benchmarks.cl_clean.analyze_cl_panel` on the imported
   JSON and check for errors/cached old-code rows;
3. append the panel block to `docs/cl_clean_sweeps.md`;
4. append only outcome-level interpretation to `docs/cl_clean_notes.md`;
5. do **not** add the result to `docs/SSUP-II.md` until the numbers have
   been independently checked against the JSON.

Panel interpretation targets:

* `accumulate_final` vs `joint`: should be equal to all stored accuracy
  digits if the sufficient-statistic identity carries through the
  streaming feature map.
* `woodbury_online` vs `accumulate_final`: tests whether real online
  decoder availability preserves the same final accuracy at practical
  batch size.
* `accumulate_task_solve_n4000`: measures the cost of making an
  accumulate-only decoder available after every task; it should be
  accurate but may be much slower than the final-only solve.
* `reset`: lower bound that sees only the final permutation.

## H8 streaming-continual diagnostic GPU panel

The first H8 panel landed as a boundary result: exact
`accumulate_final` and `joint` match, but `woodbury_online` is far
higher.  The diagnostic panel is designed to explain that mechanism, not
to chase a best configuration.

Status: **complete**.  The default diagnostic grid and optional 16k
extension produced 66 rows: 33 cells × 2 seeds.

Notebook pattern:

```text
colab/cl_clean/streaming_permuted_smnist_diagnostics.ipynb
```

Quick validation:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_diagnostics.py \
    --quick --seeds 0
```

Default diagnostic panel:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_diagnostics.py \
    --seeds 0-1
```

Optional 16k extension:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_diagnostics.py \
    --seeds 0-1 --extended
```

Imported outputs:

```text
results/cl_clean/gpu_h8_streaming_diagnostics.json
results/cl_clean/logs/gpu_h8_streaming_diagnostics.log
results/cl_clean/colab_gpu_configs/base_config_gpu_h8_streaming_diagnostics.json
results/cl_clean/cl_clean_streaming_smnist_diagnostics_bundle.zip
```

Result read:

1. `refresh_*`: exact inverse refresh collapses the high-Woodbury
   model at every width (e.g. 4000 neurons: 56.3% before refresh,
   19.4% after refresh; 16000 neurons: 59.6% before, 14.6% after).
2. `exact_*` vs `woodbury_*` over alpha: Woodbury is almost flat over
   $10^{-4}$ through $10^{-2}$; the exact solve is much more sensitive.
3. `t1`, `t2`, `t5`, `t10`: Woodbury degrades smoothly with task count;
   exact accuracy is already weak at one task under the default
   first-task bias and $\alpha = 10^{-3}$.
4. `center_first/all/zero`: zero bias recovers much of the 4000-neuron
   exact-solve gap (52.7% at $\alpha = 10^{-3}$), so first-task
   data-driven bias is part of the failure mode.
5. `bs128`, `bs512`, `reverse`: Woodbury remains 56.3% at 4000 neurons,
   so the advantage is not a tested batch-size or task-order accident.

## H8 trace-scaled-alpha repair package

Status: **ready to run**.  This supersedes the old fixed-$\alpha$ H8
streaming-continual interpretation.  The old rows remain useful as a
failure diagnostic, but they are not final Paper-II numbers because the
panel did not carry over Paper I's trace-scaled streaming regularizer.

Scope decision after audit:

* **Do not rerun the static CL panels** (B1/B3/B4/C1/C2) for this issue.
  They use `NEFLayer.solve_accumulated()`, which calls
  `solve_from_normal_equations()` and therefore applies
  $\alpha \cdot \mathrm{tr}(A^\top A) / n$.
* **Do not rerun MLP/EWC/ER baselines** for this issue.  They do not use
  NEF ridge regularization.
* **Treat §5.4 solver-mode rows carefully.**  They remain a valid tuned
  operational comparison, but `accumulate` uses trace-scaled alpha while
  Woodbury uses a fixed inverse regularizer.  They should not be used as a
  same-alpha identity proof.
* **Rerun H8 only.**  `StreamingNEFClassifier.solve()` and
  `continuous_fit()` intentionally take fixed alpha; Paper I wrapped them
  in a pilot pass to compute `alpha_eff = alpha * trace(A^T A) / n`, but
  the H8 CL runner did not.  That was the scale bug.

Stage A — mandatory repair diagnostic (choose alpha and check precision):

```bash
python -u colab/cl_clean/streaming_permuted_smnist_diagnostics.py \
    --quick --trace-repair --seeds 0

python -u colab/cl_clean/streaming_permuted_smnist_diagnostics.py \
    --trace-repair --seeds 0-1
```

The default trace-repair diagnostic runs 50 rows:

* `exact` and `woodbury` at 4000 and 8000 neurons;
* trace-scaled nominal alpha in
  `{1e-8, 1e-7, 1e-6, 1e-5, 1e-4}`;
* float32 normal-equation GEMMs for the main grid;
* float64 exact-solve normal-equation GEMMs at 4000 neurons for the same
  alpha values.

Estimated G4 time: roughly **3-5 hours** for Stage A, based on the prior
66-row fixed-alpha diagnostic taking 3.3 hours of summed row wall time.
Use the optional extension only if Stage A leaves ambiguity:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_diagnostics.py \
    --trace-repair --extended --seeds 0-1
```

The extension adds 16000-neuron trace-scaled exact/Woodbury rows and
8000-neuron float64 exact rows.  Estimated G4 time for default+extended:
roughly **5-8 hours**.

Expected Stage-A outputs:

```text
results/cl_clean/gpu_h8_streaming_trace_repair_diagnostics.json
results/cl_clean/logs/gpu_h8_streaming_trace_repair_diagnostics.log
results/cl_clean/colab_gpu_configs/base_config_gpu_h8_streaming_diagnostics.json
results/cl_clean/cl_clean_streaming_smnist_diagnostics_bundle.zip
```

Stage-A decision rule:

1. Pick the smallest trace-scaled nominal alpha whose exact solve is on
   the accuracy plateau at both 4000 and 8000 neurons.
2. If float64 exact rows materially differ from float32 exact rows at the
   chosen alpha, rerun the headline with `--stats-dtype float64`.
   Otherwise use `float32` statistics for cost.
3. Only after Stage A is imported and checked should the 5-seed headline
   panel be run.

Imported Stage-A verdict (2026-05-10): the default trace-repair grid
selects nominal `--alpha 1e-6`.  At 4000 and 8000 neurons, exact and
Woodbury agree at this alpha, and the 4000-neuron float64 exact rows do
not materially differ from float32.  **Next run the Stage-A extension**,
not the headline yet, because the planned headline uses the 16000-neuron
extension and the default Stage A did not test 16000 neurons or
8000-neuron float64 exact solves.

Next command:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_diagnostics.py \
    --trace-repair --extended --seeds 0-1
```

If the extension preserves the `1e-6` exact/Woodbury equivalence, proceed
to Stage B with `--alpha 1e-6 --alpha-scale trace --stats-dtype float32`.

Stage B — 5-seed repaired headline panel (run only after Stage A picks
`<alpha>` and `<stats_dtype>`):

```bash
python -u colab/cl_clean/streaming_permuted_smnist_gpu_suite.py \
    --quick --seeds 0 --alpha <alpha> --alpha-scale trace --stats-dtype <stats_dtype>

python -u colab/cl_clean/streaming_permuted_smnist_gpu_suite.py \
    --seeds 0-4 --extended --alpha <alpha> --alpha-scale trace --stats-dtype <stats_dtype>
```

The repaired headline phase writes a distinct JSON, e.g. for
`--alpha 1e-7 --stats-dtype float32`:

```text
results/cl_clean/gpu_h8_streaming_trace_a1em07_float32_panel.json
results/cl_clean/logs/gpu_h8_streaming_trace_a1em07_float32_panel.log
results/cl_clean/colab_gpu_configs/base_config_gpu_h8_streaming_trace_a1em07_float32.json
results/cl_clean/cl_clean_streaming_smnist_gpu_bundle.zip
```

Estimated G4 time: roughly **4-6 hours** for the 5-seed headline with the
16k extension.  Trace scaling adds a pilot accumulation pass for Woodbury
and task-solve cells, so expect it to be slower than the prior fixed-alpha
panel but still comfortably below an overnight Colab slot.

## Driver edits

**Do not edit `cl_hp_driver.sh` while a phase that uses it is in
flight.**  Bash re-reads the script file mid-execution.  If you need
to register a new panel: add it as a new `case` clause and only land
the change between phases, or copy the driver to
`run_<phase>.sh` for the next phase.

## Code-identity discipline

Every result row carries `code_identity`.  If you change anything
under `CODE_IDENTITY_PATHS` (see `cl_repro.py`), every in-flight row
becomes "old code" and the analyzer will exclude it from aggregates.
Land code changes between phases, never mid-phase.

## Reproducibility hazards (recap of AGENTS.md)

* Explicit `torch.Generator()` seeded from `_make_subrngs(seed)`.
* Encoders + gains depend only on the seed, never on the contrasted
  factor.
* Detached launches via `setsid ... < /dev/null & disown`.
* Watchdogs do not survive CLI sandbox restarts; re-arm per
  session.
* Dataset: every JSON row must carry `dataset_metadata.raw_hashes`
  and `dataset_metadata.processed_hashes`, and (for CL) the
  `stream_metadata` block emitted by `make_task_stream`.

## Hand-back protocol

When `B1..B5` and the separate Phase-C Colab bundle have all landed:

1. Tag the head of `paper-ii/cl-experiments` as
   `paper-ii/cl-data-frozen`.
2. Hand back to the scaffolding session for figure generation,
   number cross-check, and manuscript drafting.
