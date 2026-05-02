"""C5 — recurrent NEF on sMNIST (lite).

Single config × 5 seeds × 2 strategies = 10 cells.  Strategy is the
knob (``cell`` ∈ {``end_to_end``, ``target_prop``}).  Confirms the TR
§5.8 finding that only end-to-end with NEF warm-start produces
competitive results on row sMNIST; greedy and hybrid struggle
because random encoders compound state-feedback noise across
timesteps.

This is a lite phase: we do not sweep hidden width or strategy
families exhaustively.  Picking just E2E and TP shows the gap from
greedy lineage (TP) to gradient lineage (E2E) cleanly.
"""

from __future__ import annotations

import time

import torch

from leenef.recurrent import RecurrentNEFLayer

from ._phase_c_shared import (
    _make_subrngs_phase_c,
    load_dataset_for_phase_c,
    set_sl_seed,
)

C5_CELLS: tuple[str, ...] = ("end_to_end", "target_prop")


def _c5_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    strategy = str(config["cell"])
    if strategy not in C5_CELLS:
        raise ValueError(f"C5 cell {strategy!r}: must be one of {C5_CELLS}")

    n_neurons = int(config.get("n_neurons", 1500))
    d_state = int(config.get("d_state", 128))
    n_epochs = int(config.get("n_epochs", 30))
    lr = float(config.get("lr", 1e-3))
    batch_size = int(config.get("batch_size", 256))
    alpha = float(config.get("alpha", 1e-3))

    bundle = load_dataset_for_phase_c(run_context)
    x_train, y_train = bundle.x_train, bundle.y_train  # (N, T, d)
    x_test, y_test = bundle.x_test, bundle.y_test
    Y_oh = bundle.one_hot(y_train).float()

    rngs = _make_subrngs_phase_c(seed)
    init_rng = rngs["init"]

    layer = RecurrentNEFLayer(
        d_in=int(x_train.shape[2]),
        n_neurons=n_neurons,
        d_out=bundle.n_classes,
        d_state=d_state,
        activation="relu",  # recurrent default; abs explodes in BPTT
        encoder_strategy="hypersphere",
        gain=(0.5, 2.0),
        rng=init_rng,
        centers=x_train,
    )

    t0 = time.perf_counter()
    if strategy == "end_to_end":
        layer.fit_end_to_end(
            x_train,
            Y_oh,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            loss="ce",
            alpha=alpha,
        )
    elif strategy == "target_prop":
        layer.fit_target_prop(
            x_train,
            Y_oh,
            n_iters=n_epochs,
            lr=lr,
            batch_size=batch_size,
            alpha=alpha,
        )
    train_time = time.perf_counter() - t0

    @torch.no_grad()
    def _accuracy(x, y, chunk=128):
        n_correct = 0
        for s in range(0, x.shape[0], chunk):
            e = min(s + chunk, x.shape[0])
            preds = layer(x[s:e]).argmax(dim=1)
            n_correct += int((preds == y[s:e]).sum())
        return n_correct / x.shape[0]

    test_acc = _accuracy(x_test, y_test)
    train_acc = _accuracy(x_train[:5000], y_train[:5000])  # subsample, BPTT is slow

    return {
        "test_accuracy": round(test_acc, 6),
        "train_accuracy": round(train_acc, 6),
        "train_time_s": round(train_time, 4),
        "factors": {"strategy": strategy},
        "diagnostics": {
            "n_neurons": n_neurons,
            "d_state": d_state,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
        },
        "dataset_metadata": bundle.metadata(),
    }


def _c5_smnist_row_context() -> dict:
    return {
        "experiment_kind": "smnist_row_recurrent_c5_lite_panel",
        "dataset": "sMNIST-row",
        "preprocessing": "smnist_row_div255",
    }


C5_BASE_CONFIG: dict[str, object] = {
    "cell": "end_to_end",
    "n_neurons": 1500,
    "d_state": 128,
    "n_epochs": 30,
    "lr": 1e-3,
    "batch_size": 256,
    "alpha": 1e-3,
}


C5_DESCRIPTION = (
    "C5 recurrent sMNIST (lite). Knob 'cell' = strategy "
    "∈ {end_to_end, target_prop}.  Single config × 5 seeds × 2 cells.  "
    "Confirms TR §5.8: only E2E with NEF warm-start is competitive."
)
