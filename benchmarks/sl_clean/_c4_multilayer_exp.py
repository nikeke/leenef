"""C4 — multi-layer training-strategy comparison.

Knob is ``cell`` = strategy ∈ {greedy, hybrid, target_prop, end_to_end,
hybrid_e2e}.  Architecture fixed: 1 hidden layer (2000 neurons) + output
layer (2000 neurons).  This is the load-bearing **negative** result for
H6: multi-layer gradient-free stacking does not beat a single-layer
wide solve; only ``end_to_end`` (with NEF warm-start) helps modestly,
and ``hybrid_e2e`` is typically the best multi-layer variant.

Two datasets: MNIST and CIFAR-10 (Fashion-MNIST omitted to keep the
phase scope manageable; the qualitative picture is the same).
5 seeds × 5 strategies × 2 datasets = 50 cells.
"""

from __future__ import annotations

import time

import torch

from leenef.networks import NEFNetwork

from ._phase_c_shared import (
    _make_subrngs_phase_c,
    load_dataset_for_phase_c,
    set_sl_seed,
)

C4_CELLS: tuple[str, ...] = (
    "greedy",
    "hybrid",
    "target_prop",
    "end_to_end",
    "hybrid_e2e",
)


def _c4_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    strategy = str(config["cell"])
    if strategy not in C4_CELLS:
        raise ValueError(f"C4 cell {strategy!r}: must be one of {C4_CELLS}")

    hidden_neurons = list(config.get("hidden_neurons", [2000]))
    output_neurons = int(config.get("output_neurons", 2000))
    alpha = float(config["alpha"])
    n_iters = int(config.get("n_iters", 30))
    lr = float(config.get("lr", 1e-3))
    batch_size = int(config.get("batch_size", 256))

    bundle = load_dataset_for_phase_c(run_context)
    x_train, y_train = bundle.x_train, bundle.y_train
    x_test, y_test = bundle.x_test, bundle.y_test
    n_classes = bundle.n_classes
    Y_oh = bundle.one_hot(y_train).float()

    rngs = _make_subrngs_phase_c(seed)
    init_rng = rngs["init"]

    net = NEFNetwork(
        d_in=int(x_train.shape[1]),
        d_out=n_classes,
        hidden_neurons=hidden_neurons,
        output_neurons=output_neurons,
        activation="abs",
        encoder_strategy="hypersphere",
        gain=(0.5, 2.0),
        rng=init_rng,
        centers=x_train,
    )

    t0 = time.perf_counter()
    if strategy == "greedy":
        net.fit_greedy(x_train, Y_oh, alpha=alpha)
    elif strategy == "hybrid":
        net.fit_hybrid(
            x_train,
            Y_oh,
            n_iters=n_iters,
            lr=lr,
            alpha=alpha,
            batch_size=batch_size,
            loss="ce",
        )
    elif strategy == "target_prop":
        net.fit_target_prop(
            x_train,
            Y_oh,
            n_iters=n_iters,
            lr=lr,
            alpha=alpha,
            batch_size=batch_size,
        )
    elif strategy == "end_to_end":
        net.fit_end_to_end(
            x_train,
            Y_oh,
            n_iters=n_iters,
            lr=lr,
            alpha=alpha,
            batch_size=batch_size,
            loss="ce",
        )
    elif strategy == "hybrid_e2e":
        net.fit_hybrid_e2e(
            x_train,
            Y_oh,
            n_iters=n_iters,
            lr=lr,
            alpha=alpha,
            batch_size=batch_size,
            loss="ce",
        )
    train_time = time.perf_counter() - t0

    @torch.no_grad()
    def _accuracy(x, y, chunk=2000):
        n_correct = 0
        for s in range(0, x.shape[0], chunk):
            e = min(s + chunk, x.shape[0])
            preds = net(x[s:e]).argmax(dim=1)
            n_correct += int((preds == y[s:e]).sum())
        return n_correct / x.shape[0]

    train_acc = _accuracy(x_train, y_train)
    test_acc = _accuracy(x_test, y_test)

    return {
        "test_accuracy": round(test_acc, 6),
        "train_accuracy": round(train_acc, 6),
        "train_time_s": round(train_time, 4),
        "factors": {
            "strategy": strategy,
            "hidden_neurons": hidden_neurons,
            "output_neurons": output_neurons,
        },
        "diagnostics": {
            "n_iters": n_iters if strategy != "greedy" else 0,
            "lr": lr if strategy != "greedy" else 0.0,
            "batch_size": batch_size if strategy != "greedy" else 0,
        },
        "dataset_metadata": bundle.metadata(),
    }


def _c4_mnist_context() -> dict:
    return {
        "experiment_kind": "mnist_multilayer_c4_strategy_panel",
        "dataset": "MNIST",
        "preprocessing": "div255_flat",
        "n_train": 60000,
        "n_test": 10000,
        "d_in": 784,
        "n_classes": 10,
        "image_shape": [28, 28],
    }


def _c4_cifar10_context() -> dict:
    return {
        "experiment_kind": "cifar10_multilayer_c4_strategy_panel",
        "dataset": "CIFAR10",
        "preprocessing": "div255_flat",
        "n_train": 50000,
        "n_test": 10000,
        "d_in": 3072,
        "n_classes": 10,
        "image_shape": [32, 32],
    }


C4_BASE_CONFIG: dict[str, object] = {
    "cell": "greedy",
    "hidden_neurons": [2000],
    "output_neurons": 2000,
    "alpha": 1e-3,
    "n_iters": 30,
    "lr": 1e-3,
    "batch_size": 256,
}


C4_DESCRIPTION = (
    "C4 multi-layer training-strategy comparison.  Knob 'cell' = "
    "strategy ∈ {greedy, hybrid, target_prop, end_to_end, hybrid_e2e}.  "
    "Architecture: hidden=[2000], output=2000, abs, hypersphere, "
    "data_driven biases.  Reports test_accuracy and train_time."
)
