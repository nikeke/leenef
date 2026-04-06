# Running GPU Experiments in Google Colab

The repository includes a thin Colab launcher notebook and checked-in
benchmark scripts so that Colab is only the execution surface, not the
source of experiment logic.

## Recommended order

1. Open `notebooks/colab_launcher.ipynb` in Colab.
2. Run the **smoke test** first.
3. Run the **row-focused suite** next.
4. If that looks good, run the **hard sequential suite**.
5. Save the produced JSON / CSV files and share them back.

Longer suites print explicit progress now, so a cell that stays quiet for
minutes is no longer expected behaviour.

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

Provides two predefined suites:

- `row_focus` — the first recommended Colab run
  - StreamNEF on row-wise sMNIST at a fast config
  - StreamNEF on row-wise sMNIST at a stronger config
  - LSTM baseline on row-wise sMNIST
- `sequential_hard` — the longer-sequence follow-up
  - StreamNEF on pixel sMNIST
  - StreamNEF on permuted-pixel sMNIST
  - LSTM baselines on both

Both suites support `--quick` for a much smaller validation run.

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

### 3. Harder follow-up: pixel and permuted-pixel sMNIST

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

One caveat matters:

- `continuous_fit()` uses a **float64 Woodbury inverse** for stability.

That is numerically necessary, but Colab GPUs are not especially fast at
float64.  So the strongest GPU story may not be “Woodbury becomes
dramatically faster,” but rather:

- larger sequence tasks become practical,
- larger sweeps become easier to run,
- and the comparison against gradient-trained baselines becomes cleaner.

## GPU vs TPU

Prefer a **CUDA GPU** for now.

The current Colab path is plain PyTorch and does **not** include a
`torch_xla` TPU execution path.  Also, the continuous Woodbury update uses
float64 for numerical stability, which is a better fit for CPU / GPU than
for TPU-focused mixed-precision workflows.

So for the current repository state:

- **T4 / L4 GPU:** recommended
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
