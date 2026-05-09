# Split-CIFAR-10 continual ConvNEF GPU suite (Colab)

This directory contains the Paper-II **Phase C** Colab assets:

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

## Running in Colab

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
