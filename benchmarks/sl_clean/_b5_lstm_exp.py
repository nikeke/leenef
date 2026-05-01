"""B5 — LSTM baseline on temporal tasks (sMNIST-row, permuted sMNIST).

Knob ``cell`` is ``"h{hidden}_l{layers}_lr{lr}"``.  Other hyperparameters
(weight decay, batch size, epochs, patience, grad clip) are held fixed
in the base config.

The reduced grid for the headline panel is:

    hidden_size ∈ {64, 128, 256}
    num_layers  ∈ {1, 2}
    lr          ∈ {1e-3, 3e-3}

= 12 cells / dataset.
"""

from __future__ import annotations

import os
import re
import time

import torch

from .baselines.lstm import LSTMConfig, train_lstm
from .datasets import load_dataset

_CELL_RE = re.compile(r"h(\d+)_l(\d+)_lr([0-9.]+(?:e-?\d+)?)")


def _parse_cell(cell: str) -> tuple[int, int, float]:
    m = _CELL_RE.fullmatch(cell)
    if not m:
        raise ValueError(f"unrecognised B5 cell: {cell!r}")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


_B5_GRID: tuple[str, ...] = tuple(
    f"h{h}_l{lay}_lr{lr}" for h in (64, 128, 256) for lay in (1, 2) for lr in (1e-3, 3e-3)
)


def _b5_run(config: dict, seed: int, run_context: dict) -> dict:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
        torch.set_num_threads(threads)

    cell = config["cell"]
    hidden, layers, lr = _parse_cell(cell)

    cfg = LSTMConfig(
        hidden_size=hidden,
        num_layers=layers,
        lr=lr,
        weight_decay=float(config.get("weight_decay", 1e-4)),
        batch_size=int(config.get("batch_size", 128)),
        epochs=int(config.get("epochs", 30)),
        val_fraction=float(config.get("val_fraction", 0.1)),
        patience=int(config.get("patience", 5)),
        grad_clip=float(config.get("grad_clip", 1.0)),
        dropout=float(config.get("dropout", 0.0)),
    )

    bundle = load_dataset(run_context["dataset"], run_context["preprocessing"])
    meta = bundle.metadata()

    g_init = torch.Generator()
    g_init.manual_seed(seed * 7919 + 1)
    g_split = torch.Generator()
    g_split.manual_seed(seed * 7919 + 2)
    g_shuffle = torch.Generator()
    g_shuffle.manual_seed(seed * 7919 + 3)

    t0 = time.perf_counter()
    out = train_lstm(
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
            "hidden": hidden,
            "layers": layers,
            "lr": lr,
        },
        "dataset_metadata": meta,
    }


def _b5_smnist_row_context() -> dict:
    return {"dataset": "sMNIST-row", "preprocessing": "smnist_row_div255"}


def _b5_smnist_permuted_context() -> dict:
    return {
        "dataset": "sMNIST-permuted-seed0",
        "preprocessing": "smnist_permuted_div255_seed=0",
    }


B5_BASE_CONFIG: dict[str, object] = {
    "cell": _B5_GRID[0],
    "weight_decay": 1e-4,
    "batch_size": 128,
    "epochs": 30,
    "patience": 5,
    "val_fraction": 0.1,
    "grad_clip": 1.0,
    "dropout": 0.0,
}

# Permuted sMNIST is much harder; allow more epochs.
B5_PERMUTED_BASE_CONFIG: dict[str, object] = {**B5_BASE_CONFIG, "epochs": 50, "patience": 8}

B5_GRID = _B5_GRID
B5_DESCRIPTION = (
    "B5 LSTM baseline (H4/H5 honest comparison): knob 'cell' = "
    "h{hidden}_l{layers}_lr{lr}; 12-cell grid (hidden ∈ {64,128,256}, "
    "layers ∈ {1,2}, lr ∈ {1e-3, 3e-3}); fixed Adam, cosine LR, "
    "weight_decay=1e-4, batch=128, grad_clip=1.0, val_fraction=0.1.  "
    "Reports best test acc at best val epoch."
)
