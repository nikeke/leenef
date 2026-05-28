"""C6 — ConvNEF on MNIST/CIFAR-10 (lite).

Single best-known config × 5 seeds × 2 datasets = 10 cells.
Confirms TR §5.10 headline that ConvNEF matches Coates 2011-class
shallow pipelines (~78% on CIFAR-10) but not modern CNNs.

The config used here is intentionally smaller than the 80-config
sweep that produced the TR's published numbers — this phase
just regenerates rl_clean-grade SE bars on the headline
configuration so we can quote it honestly.
"""

from __future__ import annotations

import time

import torch

from leenef.conv import ConvNEFPipeline

from ._phase_c_shared import (
    load_dataset_for_phase_c,
    set_sl_seed,
)

C6_CELLS: tuple[str, ...] = ("default",)


def _c6_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    n_neurons = int(config.get("n_neurons", 4000))
    alpha = float(config.get("alpha", 1e-2))
    pool_levels = list(config.get("pool_levels", [1, 2, 4]))
    standardize = bool(config.get("standardize", True))

    bundle = load_dataset_for_phase_c(run_context)
    image_shape = tuple(run_context["image_shape"])

    if run_context["dataset"] == "MNIST":
        x_train = bundle.x_train.reshape(-1, 1, *image_shape)
        x_test = bundle.x_test.reshape(-1, 1, *image_shape)
        stages = [{"n_filters": 64, "patch_size": 5, "pool_size": 2, "filter_strategy": "pca"}]
    else:  # CIFAR10
        x_train = bundle.x_train.reshape(-1, 3, *image_shape)
        x_test = bundle.x_test.reshape(-1, 3, *image_shape)
        stages = [{"n_filters": 96, "patch_size": 5, "pool_size": 2, "filter_strategy": "pca"}]

    Y_oh = bundle.one_hot(bundle.y_train).float()

    pipe = ConvNEFPipeline(
        stages=stages,
        n_neurons=n_neurons,
        pool_levels=pool_levels,
        standardize=standardize,
        gcn=(run_context["dataset"] == "CIFAR10"),
        encoder_strategy="hypersphere",
        activation="abs",
        gain=(0.5, 2.0),
    )

    t0 = time.perf_counter()
    pipe.fit(x_train, Y_oh, alpha=(alpha, "trace"), batch_size=500, seed=seed * 9173 + 1)
    train_time = time.perf_counter() - t0

    @torch.no_grad()
    def _accuracy(x, y, chunk=500):
        n_correct = 0
        for s in range(0, x.shape[0], chunk):
            e = min(s + chunk, x.shape[0])
            preds = pipe(x[s:e]).argmax(dim=1)
            n_correct += int((preds == y[s:e]).sum())
        return n_correct / x.shape[0]

    test_acc = _accuracy(x_test, bundle.y_test)
    train_acc = _accuracy(x_train[:5000], bundle.y_train[:5000])

    return {
        "test_accuracy": round(test_acc, 6),
        "train_accuracy": round(train_acc, 6),
        "train_time_s": round(train_time, 4),
        "factors": {"config": "default"},
        "diagnostics": {
            "n_neurons": n_neurons,
            "stages": stages,
            "pool_levels": pool_levels,
            "standardize": standardize,
        },
        "dataset_metadata": bundle.metadata(),
    }


def _c6_mnist_context() -> dict:
    return {
        "experiment_kind": "mnist_convnef_c6_lite_panel",
        "dataset": "MNIST",
        "preprocessing": "div255_flat",
        "n_train": 60000,
        "n_test": 10000,
        "image_shape": [28, 28],
        "n_classes": 10,
    }


def _c6_cifar10_context() -> dict:
    return {
        "experiment_kind": "cifar10_convnef_c6_lite_panel",
        "dataset": "CIFAR10",
        "preprocessing": "div255_flat",
        "n_train": 50000,
        "n_test": 10000,
        "image_shape": [32, 32],
        "n_classes": 10,
    }


C6_BASE_CONFIG: dict[str, object] = {
    "cell": "default",
    "n_neurons": 4000,
    "alpha": 1e-2,
    "pool_levels": [1, 2, 4],
    "standardize": True,
}


C6_DESCRIPTION = (
    "C6 ConvNEF (lite). Single 'default' cell × 5 seeds × {MNIST, CIFAR-10}.  "
    "Confirms TR §5.10 headline within rl_clean SE bars."
)
