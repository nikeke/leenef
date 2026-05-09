# Paper-II continual GPU suites (Colab)

This directory contains two Paper-II Colab packages:

1. the Split-CIFAR-10 ConvNEF continual panel used in the current
   manuscript; and
2. the H8 streaming-continual follow-up panel, designed but not yet part
   of the manuscript claims.

## Split-CIFAR-10 ConvNEF suite

| Panel | Experiment | Cells | Seeds | Why it is included |
|------|------------|-------|-------|--------------------|
| `gpu_c1_convnef_split_cifar10_panel` | `c1_convnef_split_cifar10` | `all_data_n10000`, `first_task_n10000`, `joint_n10000`, `all_data_n20000`, `first_task_n20000`, `joint_n20000` | `0-9` | The actual ConvNEF-CL visual-extension panel. |
| `gpu_c2_flat_nef_split_cifar10_panel` | `nef_continual_split_cifar10` | `nef_accumulate` | `0-9` | The flat-NEF floor on the same device and same code identity. |

The runner is `continual_cifar10_gpu_suite.py`.  It uses the committed
benchmark code under `benchmarks/cl_clean/` and the standard
`benchmarks.cl_clean.cl_phase_runner`.  There is **no runtime
monkey-patching**.

## Code identity

The old CL Colab suite kept the core logic in a standalone benchmark
script.  The `cl_clean` version fixes that: the Phase-C experiment now
lives under `benchmarks/cl_clean/_c1_convnef_cl_exp.py` and is
registered as `c1_convnef_split_cifar10`.

That matters because:

1. every GPU row now carries the normal `cl_clean` `code_identity`;
2. GPU rows store `"device": "cuda"` in their config, so they cannot be
   confused with CPU rows;
3. the notebook only clones a committed branch, rewrites GPU configs,
   runs the normal phase runner, and bundles artifacts.

## GPU recommendation

Per `AGENTS.md`, **G4** is preferred if Colab offers it at the usual
~5 CU/hr rate.  **L4** is the fallback.  The suite fits comfortably on
either card; A100/H100 are unnecessary here.

## Running the ConvNEF suite in Colab

1. Push this branch, or edit the notebook's `BRANCH` variable to a
   pushed ref that contains these files.
2. Open `continual_cifar10_gpu_suite.ipynb`.
3. Runtime -> Change runtime type -> GPU -> **G4** if available,
   otherwise **L4**.
4. Run all cells.
5. Download `cl_clean_phase_c_gpu_bundle.zip` from the final cell and
   copy the JSON, logs, and GPU config snapshots back into this branch.

By default the notebook clones **`paper-ii/cl-scaffold`**, because the
Phase-C package is committed here rather than on
`paper-ii/cl-experiments`.

## H8 streaming continual suite

The H8 suite is:

```text
streaming_permuted_smnist_gpu_suite.ipynb
streaming_permuted_smnist_gpu_suite.py
```

It runs `h8_streaming_permuted_smnist` through the standard phase runner
on a 10-task permuted sequential-MNIST stream.  The default cells are:

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

Default command:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_gpu_suite.py --seeds 0-4
```

Quick validation:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_gpu_suite.py --quick --seeds 0
```

Optional extension:

```bash
python -u colab/cl_clean/streaming_permuted_smnist_gpu_suite.py --seeds 0-4 --extended
```

Use G4 if available, L4 fallback.  The suite is expected to be too slow
for CPU because it multiplies the already-expensive permuted-sMNIST
streaming solve by 10 task permutations and multiple solver modes.

## Quick validation

For a short smoke run before the full 10-seed panel:

```bash
python -u colab/cl_clean/continual_cifar10_gpu_suite.py --quick --seeds 0
```

That runs a single `all_data_n2000` ConvNEF cell on the GPU and still
produces a bundle.

## Optional CPU mirror

The default suite is GPU-only.  If we later need a same-code-identity
CPU mirror for the speedup table, rerun the same experiments with a CPU
config copy (`"device": "cpu"`) and commit the resulting rows
separately.
