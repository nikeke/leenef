# Paper III sparse-reward GPU suite (Colab)

This directory contains the Colab package for sparse-reward successor-feature
experiments in Paper III.

The suite runs the committed `benchmarks/run_rl_sparse.py` runner with
`--device cuda`; there is no runtime monkey-patching. It is focused on the
directional successor readout because that is the lowest-memory path for the
known-goal MountainCar and Acrobot panels.

## GPU recommendation

Use **G4** if Colab offers it at the expected rate. **L4** is the fallback.
A100/H100 are useful only if the float64 Cholesky solves, not memory capacity,
become the bottleneck.

## Running in Colab

1. Push a branch that contains this directory.
2. Open `sparse_successor_gpu_suite.ipynb`.
3. Runtime -> Change runtime type -> GPU -> G4 if available, otherwise L4.
4. Run the CUDA check, clone/setup, install, run, and download cells.
5. Copy the downloaded `rl_sparse_gpu_bundle.zip` contents back into this branch.

Quick validation:

```bash
python -u colab/rl_sparse/sparse_successor_gpu_suite.py --quick --seeds 0 --width 512 --episodes 5
```

Default full-width directional panel:

```bash
python -u colab/rl_sparse/sparse_successor_gpu_suite.py --seeds 0-9 --width 6000
```

Optional Acrobot retention scan:

```bash
python -u colab/rl_sparse/sparse_successor_gpu_suite.py --seeds 0-4 --width 6000 --include-scan
```
