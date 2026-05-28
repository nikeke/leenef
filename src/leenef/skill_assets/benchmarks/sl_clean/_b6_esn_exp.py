"""B6 — ESN baseline on temporal tasks (sMNIST-row, permuted sMNIST).

Knob ``cell`` is ``"n{n}_sr{sr}_leak{leak}"``.  Other hyperparameters
(input scale, sparsity, ridge α, chunk size) are held fixed in the
base config.

Headline grid:

    n_reservoir     ∈ {500, 1000, 2000}
    spectral_radius ∈ {0.7, 0.95}
    leak_rate       ∈ {0.3, 1.0}

= 12 cells / dataset, mirroring the B5 LSTM grid in size.
"""

from __future__ import annotations

import os
import re
import time

import torch

from .baselines.esn import ESNConfig, train_esn
from .datasets import load_dataset

_CELL_RE = re.compile(r"n(\d+)_sr([0-9.]+)_leak([0-9.]+)")


def _parse_cell(cell: str) -> tuple[int, float, float]:
    m = _CELL_RE.fullmatch(cell)
    if not m:
        raise ValueError(f"unrecognised B6 cell: {cell!r}")
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


_B6_GRID: tuple[str, ...] = tuple(
    f"n{n}_sr{sr}_leak{leak}"
    for n in (500, 1000, 2000)
    for sr in (0.7, 0.95)
    for leak in (0.3, 1.0)
)


def _resolve_device(config: dict) -> str:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("B6 requested CUDA but torch.cuda.is_available() is false")
    return requested


def _b6_run(config: dict, seed: int, run_context: dict) -> dict:
    device = _resolve_device(config)
    if device == "cpu":
        device = "cpu"
        threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        torch.set_num_threads(threads)

    cell = config["cell"]
    n_res, sr, leak = _parse_cell(cell)

    cfg = ESNConfig(
        n_reservoir=n_res,
        spectral_radius=sr,
        leak_rate=leak,
        input_scale=float(config.get("input_scale", 1.0)),
        sparsity=float(config.get("sparsity", 0.1)),
        ridge_alpha=float(config.get("ridge_alpha", 1e-3)),
        chunk_size=int(config.get("chunk_size", 256)),
        use_bias_state=bool(config.get("use_bias_state", True)),
    )

    bundle = load_dataset(run_context["dataset"], run_context["preprocessing"])
    meta = bundle.metadata()

    g_input = torch.Generator()
    g_input.manual_seed(seed * 7919 + 11)
    g_reservoir = torch.Generator()
    g_reservoir.manual_seed(seed * 7919 + 12)

    t0 = time.perf_counter()
    out = train_esn(
        bundle.x_train,
        bundle.y_train,
        bundle.x_test,
        bundle.y_test,
        cfg,
        generator_input=g_input,
        generator_reservoir=g_reservoir,
        device=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    return {
        "test_accuracy": out["test_accuracy"],
        "diagnostics": {
            "train_accuracy": out["train_accuracy"],
            "build_time_sec": out["build_time_sec"],
            "collect_time_sec": out["collect_time_sec"],
            "solve_time_sec": out["solve_time_sec"],
            "fit_time_sec": out["fit_time_sec"],
            "wall_sec": wall,
            "n_params": out["n_params"],
            "n_reservoir": n_res,
            "spectral_radius": sr,
            "leak_rate": leak,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        },
        "dataset_metadata": meta,
    }


def _b6_smnist_row_context() -> dict:
    return {"dataset": "sMNIST-row", "preprocessing": "smnist_row_div255"}


def _b6_smnist_permuted_context() -> dict:
    return {
        "dataset": "sMNIST-permuted-seed0",
        "preprocessing": "smnist_permuted_div255_seed=0",
    }


B6_BASE_CONFIG: dict[str, object] = {
    "cell": _B6_GRID[0],
    "input_scale": 1.0,
    "sparsity": 0.1,
    "ridge_alpha": 1e-3,
    "chunk_size": 256,
    "use_bias_state": True,
}

B6_GRID = _B6_GRID
B6_DESCRIPTION = (
    "B6 ESN baseline (H4/H5 honest comparison): knob 'cell' = "
    "n{n}_sr{sr}_leak{leak}; 12-cell grid (n ∈ {500,1000,2000}, "
    "sr ∈ {0.7,0.95}, leak ∈ {0.3,1.0}); leaky-integrator reservoir "
    "with sparse random W, spectral-radius rescaling, tanh activation, "
    "ridge readout (α=1e-3) on the final state. Gradient-free; same "
    "feature-extract + analytic-solve template as streaming-NEF."
)
