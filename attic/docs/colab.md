# Running GPU Experiments in Google Colab

The repository includes a thin Colab launcher notebook and checked-in
benchmark scripts so that Colab is only the execution surface, not the
source of experiment logic.

## Recommended order

1. Open `notebooks/colab_launcher.ipynb` in Colab.
2. Run the **smoke test** first.
3. Run the **row-focused suite** next.
4. If you are updating the Section 5.8 recurrent TP analysis, run the
   **recurrent predictive-state suite**.
5. If that looks good, run the **hard sequential suite**.
6. Save the produced JSON / CSV files and share them back.

Longer suites print explicit progress now, so a cell that stays quiet for
minutes is no longer expected behavior.

## What each artifact does

### `notebooks/colab_launcher.ipynb`

The notebook:

- clones the repository,
- installs dependencies,
- optionally mounts Google Drive,
- runs `benchmarks/colab_smoke.py`,
- then runs `benchmarks/colab_suites.py`.

Use it as the default Colab entrypoint.

### `benchmarks/colab_smoke.py`

A quick validation pass that checks:

- package import and installation,
- dataset download,
- selected device availability,
- a tiny MNIST `NEFLayer` run,
- a tiny row-wise sMNIST `StreamingNEFClassifier` run,
- JSON result persistence.

If this fails, fix the environment before attempting longer suites.

### `benchmarks/colab_suites.py`

Provides several predefined suites.  The main sequential ones are:

- `row_focus` — the first recommended Colab run
  - StreamNEF 2k (Woodbury and accumulate) on row-wise sMNIST
  - StreamNEF 8k (Woodbury and accumulate) on row-wise sMNIST
  - LSTM baseline on row-wise sMNIST
- `recurrent_predictive` — full-dataset recurrent TP comparison for Section 5.8
  - row-wise sMNIST, `RecNEF-target_prop`, `state_target=reconstruction`
  - row-wise sMNIST, `RecNEF-target_prop`, `state_target=predictive`
  - default full run uses seeds `0, 1, 2` (or `seed, seed+1, seed+2`)
  - prints per-target mean summaries after the seed sweep completes
- `sequential_hard` — the longer-sequence follow-up
  - StreamNEF on pixel sMNIST (accumulate, GPU-friendly)
  - StreamNEF on permuted-pixel sMNIST (accumulate)
  - LSTM baselines on both

All suites support `--quick` for a much smaller validation run.

The suite runner prints:

- benchmark start / finish lines,
- streaming batch progress for long StreamNEF runs,
- and epoch progress for LSTM runs.

## Suggested first Colab runs

### 1. Smoke test

Run this first from the notebook or directly:

```bash
python benchmarks/colab_smoke.py --device auto --output ./results/colab/colab_smoke.json
```

### 2. First real benchmark: row-focused suite

This should be the first benchmark suite you run on Colab:

```bash
python benchmarks/colab_suites.py \
  --suite row_focus \
  --device auto \
  --output-dir ./results/colab
```

This is the best first test because:

- we already know row-wise sMNIST is promising,
- it lets us compare StreamNEF to LSTM quickly,
- and it tells us whether Colab actually improves wall-clock time.

### 3. Section 5.8 follow-up: full-dataset recurrent predictive-state sweep

```bash
python benchmarks/colab_suites.py \
  --suite recurrent_predictive \
  --device auto \
  --output-dir ./results/colab
```

This suite exists specifically to replace the current seeded-slice
Section 5.8 result with a full-dataset recurrent TP comparison.  The
default full run benchmarks both state targets across seeds `0, 1, 2`,
flushes JSON / CSV results after every finished seed run, and prints the
mean reconstruction vs predictive accuracy at the end.

### 4. Harder follow-up: pixel and permuted-pixel sMNIST

```bash
python benchmarks/colab_suites.py \
  --suite sequential_hard \
  --device auto \
  --output-dir ./results/colab
```

This is the main “hard task” suite for Colab.  It tests whether the
streaming delay-line approach scales to much longer sequences (`T=784`)
and whether it stays competitive against an LSTM baseline.

## Interpreting the likely outcome

The `row_focus` suite now runs both Woodbury (float64) and accumulate
(float32) variants for each configuration, making the dtype impact
directly measurable.

Key caveats:

- `continuous_fit()` uses a **float64 Woodbury inverse** for stability.
  Consumer GPUs deliver only ~1/32 of float32 throughput in float64.
- `accumulate()` + `solve()` is **pure float32**, avoiding the GPU
  bottleneck.  It produces mathematically equivalent results but cannot
  provide online decoder updates during training.

The strongest GPU story is the accumulate path: it should show
significantly better speedups than the Woodbury path, especially at
8000 neurons where the float64 inverse dominates wall-clock time.

## GPU vs TPU

Prefer a **CUDA GPU** for now.

The current Colab path is plain PyTorch and does **not** include a
`torch_xla` TPU execution path.  The accumulate+solve path is pure
float32 and should work well on any GPU.  The continuous Woodbury path
uses float64, which is a better fit for CPU / GPU than for TPU.

So for the current repository state:

- **T4 / L4 GPU:** recommended (use `--suite row_focus` for both
  Woodbury and accumulate comparison)
- **TPU v5e:** not currently a priority target

## Comparing results later

To summarise one or more result files:

```bash
python benchmarks/colab_compare.py ./results/colab/*.json
```

To compare a new run against a baseline:

```bash
python benchmarks/colab_compare.py \
  --baseline ./results/colab/row_focus_cpu.json \
  ./results/colab/row_focus_gpu.json
```
