"""B4 — MLP baseline experiment registered into ``benchmarks.sl_clean``.

The knob is ``cell``, a compact string of the form ``"d{depth}_w{width}_lr{lr}"``
that encodes the MLP configuration.  Other hyperparameters (weight
decay, batch size, epochs, patience, activation) live in the base
config and are held fixed across the grid.

We rely on explicit ``torch.Generator`` instances for every stochastic
step (weight init, train/val split, per-epoch shuffling) so the
result is reproducible regardless of the global RNG state.
"""

from __future__ import annotations

import os
import time

import torch

from .baselines.mlp import MLPConfig, train_mlp
from .datasets import load_dataset

# Headline grid for the B4 sweep.  Kept small but defensible:
#   depth ∈ {1, 2}
#   width ∈ {256, 512, 1024}
#   lr    ∈ {1e-3, 3e-3}
# = 12 cells / dataset.
_B4_GRID: tuple[str, ...] = tuple(
    f"d{d}_w{w}_lr{lr}" for d in (1, 2) for w in (256, 512, 1024) for lr in (1e-3, 3e-3)
)


def _parse_cell(cell: str) -> tuple[int, int, float]:
    """Parse "d{depth}_w{width}_lr{lr}" into typed fields."""
    parts = cell.split("_")
    if (
        len(parts) != 3
        or not parts[0].startswith("d")
        or not parts[1].startswith("w")
        or not parts[2].startswith("lr")
    ):
        raise ValueError(f"unrecognised B4 cell: {cell!r}")
    depth = int(parts[0][1:])
    width = int(parts[1][1:])
    lr = float(parts[2][2:])
    return depth, width, lr


def _b4_run(config: dict, seed: int, run_context: dict) -> dict:
    """Train one MLP on (cell, seed) and return metrics."""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        # respect OMP_NUM_THREADS if set externally; default to 4.
        threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        torch.set_num_threads(threads)

    cell = config["cell"]
    depth, width, lr = _parse_cell(cell)

    cfg = MLPConfig(
        depth=depth,
        width=width,
        activation=str(config.get("activation", "relu")),
        lr=lr,
        weight_decay=float(config.get("weight_decay", 1e-4)),
        batch_size=int(config.get("batch_size", 128)),
        epochs=int(config.get("epochs", 50)),
        val_fraction=float(config.get("val_fraction", 0.1)),
        patience=int(config.get("patience", 5)),
        dropout=float(config.get("dropout", 0.0)),
    )

    bundle = load_dataset(run_context["dataset"], run_context["preprocessing"])
    meta = bundle.metadata()

    # Per-cell sub-generators derived deterministically from the seed.
    # Using independent streams isolates init / split / shuffle stochasticity
    # so that drift in one source can be diagnosed in isolation later.
    g_init = torch.Generator()
    g_init.manual_seed(seed * 7919 + 1)
    g_split = torch.Generator()
    g_split.manual_seed(seed * 7919 + 2)
    g_shuffle = torch.Generator()
    g_shuffle.manual_seed(seed * 7919 + 3)

    t0 = time.perf_counter()
    out = train_mlp(
        bundle.x_train,
        bundle.y_train,
        bundle.x_test,
        bundle.y_test,
        cfg,
        generator_init=g_init,
        generator_split=g_split,
        generator_shuffle=g_shuffle,
        device=device,
    )
    wall = time.perf_counter() - t0

    return {
        "test_accuracy": out["test_accuracy"],
        "diagnostics": {
            "val_accuracy": out["val_accuracy"],
            "train_accuracy": out["train_accuracy"],
            "best_epoch": out["best_epoch"],
            "epochs_run": out["epochs_run"],
            "fit_time_sec": out["fit_time_sec"],
            "wall_sec": wall,
            "n_params": out["n_params"],
            "depth": depth,
            "width": width,
            "lr": lr,
        },
        "dataset_metadata": {
            "dataset": meta["dataset"],
            "preprocessing": meta["preprocessing"],
            "n_train": meta["n_train"],
            "n_test": meta["n_test"],
            "d_in": meta.get("d_in", int(bundle.x_train.shape[-1])),
            "n_classes": meta["n_classes"],
            "raw_hashes": meta.get("raw_hashes", {}),
            "processed_hashes": meta.get("processed_hashes", {}),
        },
    }


def _b4_mnist_context() -> dict:
    return {"dataset": "MNIST", "preprocessing": "div255_flat"}


def _b4_fashion_context() -> dict:
    return {"dataset": "FashionMNIST", "preprocessing": "div255_flat"}


def _b4_cifar10_context() -> dict:
    return {"dataset": "CIFAR10", "preprocessing": "div255_flat"}


B4_BASE_CONFIG: dict[str, object] = {
    "cell": _B4_GRID[0],
    "activation": "relu",
    "weight_decay": 1e-4,
    "batch_size": 128,
    "epochs": 50,
    "patience": 5,
    "val_fraction": 0.1,
    "dropout": 0.0,
}


B4_GRID = _B4_GRID
B4_DESCRIPTION = (
    "B4 MLP baseline (H2 honest comparison): knob 'cell' = "
    "d{depth}_w{width}_lr{lr}; 12-cell grid (depth ∈ {1,2}, "
    "width ∈ {256,512,1024}, lr ∈ {1e-3, 3e-3}); fixed Adam, "
    "cosine LR, weight_decay=1e-4, batch=128, epochs=50, patience=5, "
    "val_fraction=0.1.  Reports best test acc at best val epoch."
)
