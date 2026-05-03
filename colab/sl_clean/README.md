# Recurrent sMNIST GPU suite (Colab)

This directory contains the Colab assets for the only Paper-I supervised
panel that is not practical on the reference CPU box: **C5 recurrent NEF
on row sMNIST**.  It also reruns the row-sMNIST comparators that make the
C5 numbers cleanly interpretable on the same GPU:

| Panel | Cells | Seeds | Why it is included |
|------|-------|-------|--------------------|
| `gpu_c5_recurrent_smnist_panel` | `end_to_end`, `target_prop` | 0-4 | The C5 recurrent NEF result itself. |
| `gpu_streaming_smnist_row_panel` | Stream-NEF `n=4000,8000` | 0-4 | Same dataset, analytic/streaming reference point. |
| `gpu_b5_lstm_smnist_row_pareto_panel` | LSTM `h256_l2_lr0.003`, `h64_l2_lr0.003` | 0-4 | Best-accuracy and fast near-best CPU-tuned LSTM cells from B5, rerun on the same GPU. |
| `gpu_b6_esn_smnist_row_best_panel` | ESN `n2000_sr0.95_leak0.3` | 0-4 | Best CPU-tuned ESN row cell from B6, rerun on the same GPU. |

The runner is `recurrent_smnist_gpu_suite.py`.  It uses the committed
benchmark code and the standard `benchmarks.sl_clean.sl_phase_runner`.
There is **no runtime monkey-patching**.

## Code identity

The previous Colab draft tried to keep C5 out of the watched
`CODE_IDENTITY_PATHS` by monkey-patching from `colab/`.  That was wrong:
the patch would not survive the subprocess boundary into
`sl_phase_runner`, and even if it did, it would be a bad identity story.

The fixed design is explicit:

1. `benchmarks/sl_clean/_c5_recurrent_lite_exp.py` has committed device
   support and its base config says `"device": "cuda"`.
2. The row comparators have committed optional `"device": "cuda"`
   support.
3. The Colab runner writes GPU-specific base-config copies under
   `results/sl_clean/colab_gpu_configs/` and invokes the normal runner.
4. Every GPU result row stores `"device": "cuda"` in its `config` and
   carries the normal `code_identity` computed from committed source.

This means GPU rows cannot be mistaken for CPU rows, and a later clone
of the same committed tree can reproduce the identity.

## GPU recommendation

Per `AGENTS.md`, **L4** is the recommended Colab GPU for this suite.
T4 is likely sufficient but slower; A100/H100 are unnecessary at these
model sizes.

Expected wall time on L4: **a few GPU hours total**.  The C5 TP cells are
the slowest; the Stream-NEF, LSTM Pareto, and ESN-best comparator panels
are included because they are small enough to rerun cleanly on the same
device.

## Running in Colab

1. Open `recurrent_smnist_gpu_suite.ipynb`.
2. Runtime -> Change runtime type -> GPU -> L4.
3. Run all cells.
4. Download the four JSON files printed by the final cell and commit them
   back to the Phase-C branch.

The notebook clones `paper-i/phase-c-experiments`, so that branch must
contain this GPU-suite commit before the notebook is run.

## What this suite does *not* replace

The CPU B2/B5/B6 panels remain the basis for the wall-clock comparisons
in §6.2 of the paper.  The GPU suite is for a clean C5 recurrent panel
and same-device row-sMNIST sanity comparisons.  It is **not** a
replacement for the in-flight CPU B6 permuted-sMNIST ESN run, whose CPU
timing is needed for the Stream-NEF-vs-ESN wall-clock claim.
