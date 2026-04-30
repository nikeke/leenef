"""Experiment registry for ``benchmarks.sl_clean``.

Each entry maps an experiment name to:

* ``run_fn(config, seed, run_context) -> dict``: produce the metric
  fields for one ``(config, seed)`` cell.  The runner adds the
  reproducibility metadata and the sweep-axis fields (``knob``,
  ``value``) on top.
* ``default_run_context()``: the canonical run context for that
  experiment (e.g. dataset, n_train, n_test).  Two cells can only be
  compared when their run contexts match.
* ``primary_metric`` / ``higher_is_better``: drives the analyzer's
  adoption decision and per-metric printing.

The first entry is a tiny synthetic smoke experiment used by the unit
tests and as the worked example in the documentation.  Headline-claim
experiments are added below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from leenef.encoders import make_encoders
from leenef.layers import NEFLayer

from .datasets import load_mnist
from .sl_repro import set_sl_seed

RunFn = Callable[[dict, int, dict], dict]
ContextFn = Callable[[], dict]


@dataclass(frozen=True)
class ExperimentSpec:
    """A registered SL clean experiment."""

    name: str
    run_fn: RunFn
    default_run_context: ContextFn
    primary_metric: str
    higher_is_better: bool = True
    description: str = ""


_REGISTRY: dict[str, ExperimentSpec] = {}


def register(spec: ExperimentSpec) -> ExperimentSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"experiment {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ExperimentSpec:
    if name not in _REGISTRY:
        raise KeyError(f"experiment {name!r} not registered (have: {sorted(_REGISTRY)})")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Smoke experiment: tiny synthetic regression / binary classification.
# Purpose: keep the runner end-to-end tested in CI without pulling MNIST.
# ---------------------------------------------------------------------------


def _smoke_make_data(n: int, dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    # Linearly separable target: sign of a fixed projection plus a small bias.
    w = torch.randn(dim, generator=g)
    logits = x @ w
    y = (logits > 0).long()
    one_hot = torch.zeros(n, 2)
    one_hot[torch.arange(n), y] = 1.0
    return x, one_hot


def _smoke_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    n_train = int(run_context.get("n_train", 256))
    n_test = int(run_context.get("n_test", 256))
    dim = int(run_context.get("dim", 8))

    x_train, y_train = _smoke_make_data(n_train, dim, seed=seed)
    x_test, y_test = _smoke_make_data(n_test, dim, seed=seed + 1_000_000)

    layer = NEFLayer(
        d_in=dim,
        n_neurons=int(config["n_neurons"]),
        d_out=2,
        encoder_strategy=str(config["encoder_strategy"]),
        activation=str(config["activation"]),
        gain=(float(config["gain_low"]), float(config["gain_high"])),
        centers=x_train if bool(config["data_driven_biases"]) else None,
        rng=torch.Generator().manual_seed(seed),
    )
    t0 = time.perf_counter()
    layer.fit(x_train, y_train, alpha=float(config["alpha"]))
    fit_s = time.perf_counter() - t0

    with torch.no_grad():
        train_pred = layer(x_train).argmax(dim=1)
        test_pred = layer(x_test).argmax(dim=1)
        train_acc = float((train_pred == y_train.argmax(dim=1)).float().mean())
        test_acc = float((test_pred == y_test.argmax(dim=1)).float().mean())

    return {
        "train_accuracy": round(train_acc, 6),
        "test_accuracy": round(test_acc, 6),
        "fit_time_s": round(fit_s, 4),
    }


def _smoke_context() -> dict:
    return {
        "experiment_kind": "synthetic_binary_classification",
        "n_train": 256,
        "n_test": 256,
        "dim": 8,
    }


SMOKE_BASE_CONFIG: dict[str, object] = {
    "n_neurons": 200,
    "encoder_strategy": "hypersphere",
    "activation": "abs",
    "gain_low": 0.5,
    "gain_high": 2.0,
    "alpha": 1e-3,
    "data_driven_biases": True,
}


register(
    ExperimentSpec(
        name="smoke",
        run_fn=_smoke_run,
        default_run_context=_smoke_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=(
            "Tiny synthetic binary-classification smoke experiment used by "
            "unit tests and as the worked example in docs/sl_clean_sweeps.md."
        ),
    )
)


# ---------------------------------------------------------------------------
# B1 — biases_mnist: H1 panel "data-driven biases close the encoder-type gap".
#
# Single factorial phase.  The knob is ``cell``, whose values enumerate
# 9 cells = ``bias_mode ∈ {data_driven, random_normal, zero}`` ×
# ``encoder_strategy ∈ {hypersphere, gaussian, sparse}``.
#
# The headline H1 panel is the 2×3 ``{data_driven, random_normal} ×
# encoder``.  ``zero`` is reported as an appendix diagnostic.
#
# Implementation notes (per rubber-duck and GPT-5.5 review):
#
# * The ``encoder`` and ``gain`` sub-generators are derived from the
#   cell-level seed *independently* of ``bias_mode``.  That means
#   encoders and gains are bit-identical across the three bias modes
#   for the same seed, so the bias-mode comparison is cleanly causal.
# * The ``center_index`` and ``random_bias`` sub-generators are *only*
#   advanced inside the corresponding bias-mode branch.
# * The accumulator path computes ``AᵀA`` and ``AᵀY`` chunk by chunk
#   via ``addmm_``, then solves with Cholesky.  This matches the
#   numerics of ``leenef.solvers.normal_equations`` (trace-scaled
#   Tikhonov regularization) but avoids materializing the full
#   ``60000×n_neurons`` activity matrix.
# ---------------------------------------------------------------------------


_CELL_SEPARATOR = "+"
_BIAS_MODES = ("data_driven", "random_normal", "zero")
_ENCODER_STRATEGIES = ("hypersphere", "gaussian", "sparse")
BIASES_MNIST_CELLS: tuple[str, ...] = tuple(
    f"{bm}{_CELL_SEPARATOR}{es}" for bm in _BIAS_MODES for es in _ENCODER_STRATEGIES
)


def _parse_biases_cell(cell: str) -> tuple[str, str]:
    if _CELL_SEPARATOR not in cell:
        raise ValueError(f"cell {cell!r}: expected 'bias_mode{_CELL_SEPARATOR}encoder_strategy'")
    bm, es = cell.split(_CELL_SEPARATOR, 1)
    if bm not in _BIAS_MODES:
        raise ValueError(f"cell {cell!r}: unknown bias_mode {bm!r}")
    if es not in _ENCODER_STRATEGIES:
        raise ValueError(f"cell {cell!r}: unknown encoder_strategy {es!r}")
    return bm, es


def _make_subrngs(seed: int) -> dict[str, torch.Generator]:
    """Independent torch.Generators for each RNG stream.

    Stride of ~1e6 between streams keeps neighboring seeds from
    interfering even if the underlying PRNG advances unevenly.
    """
    streams = ("encoder", "gain", "center_index", "random_bias")
    return {
        name: torch.Generator().manual_seed(seed * 1_000_003 + 17 * i + 1)
        for i, name in enumerate(streams)
    }


def _biases_mnist_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    bias_mode, encoder_strategy = _parse_biases_cell(str(config["cell"]))
    n_neurons = int(config["n_neurons"])
    alpha = float(config["alpha"])
    gain_low = float(config["gain_low"])
    gain_high = float(config["gain_high"])
    chunk_size = int(config.get("chunk_size", 5000))

    bundle = load_mnist(preprocessing=str(run_context.get("preprocessing", "div255_flat")))
    x_train, y_train = bundle.x_train, bundle.y_train
    x_test, y_test = bundle.x_test, bundle.y_test
    d_in = int(x_train.shape[1])
    n_classes = bundle.n_classes

    rngs = _make_subrngs(seed)

    enc = make_encoders(n_neurons, d_in, strategy=encoder_strategy, rng=rngs["encoder"])
    gain = gain_low + (gain_high - gain_low) * torch.rand(n_neurons, generator=rngs["gain"])

    if bias_mode == "data_driven":
        idx = torch.randint(len(x_train), (n_neurons,), generator=rngs["center_index"])
        bias = -gain * (x_train[idx] * enc).sum(dim=1)
    elif bias_mode == "random_normal":
        bias = torch.randn(n_neurons, generator=rngs["random_bias"])
    elif bias_mode == "zero":
        bias = torch.zeros(n_neurons)
    else:  # pragma: no cover — guarded by _parse_biases_cell
        raise ValueError(bias_mode)

    Y_oh = bundle.one_hot(y_train)
    ATA = torch.zeros(n_neurons, n_neurons, dtype=torch.float32)
    ATY = torch.zeros(n_neurons, n_classes, dtype=torch.float32)

    t0 = time.perf_counter()
    for s in range(0, x_train.shape[0], chunk_size):
        e = min(s + chunk_size, x_train.shape[0])
        z = x_train[s:e] @ enc.T
        z.mul_(gain)
        z.add_(bias)
        z.abs_()
        ATA.addmm_(z.T, z)
        ATY.addmm_(z.T, Y_oh[s:e])
    encode_fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    reg = alpha * torch.trace(ATA) / n_neurons
    ATA.diagonal().add_(torch.clamp(reg, min=alpha))
    L = torch.linalg.cholesky(ATA)
    D = torch.cholesky_solve(ATY, L)
    solve_s = time.perf_counter() - t0

    def _accuracy(x_split: torch.Tensor, y_split: torch.Tensor) -> float:
        n_correct = 0
        for s in range(0, x_split.shape[0], chunk_size):
            e = min(s + chunk_size, x_split.shape[0])
            z = x_split[s:e] @ enc.T
            z.mul_(gain)
            z.add_(bias)
            z.abs_()
            preds = (z @ D).argmax(dim=1)
            n_correct += int((preds == y_split[s:e]).sum())
        return n_correct / x_split.shape[0]

    train_acc = _accuracy(x_train, y_train)
    test_acc = _accuracy(x_test, y_test)

    enc_norms = enc.norm(dim=1)

    return {
        "test_accuracy": round(test_acc, 6),
        "train_accuracy": round(train_acc, 6),
        "encode_fit_time_s": round(encode_fit_s, 4),
        "solve_time_s": round(solve_s, 4),
        "factors": {"bias_mode": bias_mode, "encoder_strategy": encoder_strategy},
        "diagnostics": {
            "encoder_norm_mean": round(float(enc_norms.mean()), 6),
            "encoder_norm_std": round(float(enc_norms.std()), 6),
            "bias_mean": round(float(bias.mean()), 6),
            "bias_std": round(float(bias.std()), 6),
            "gain_mean": round(float(gain.mean()), 6),
        },
        "dataset_metadata": bundle.metadata(),
    }


def _biases_mnist_context() -> dict:
    return {
        "experiment_kind": "mnist_singlelayer_biases_panel",
        "dataset": "MNIST",
        "preprocessing": "div255_flat",
        "n_train": 60000,
        "n_test": 10000,
        "d_in": 784,
        "n_classes": 10,
    }


BIASES_MNIST_BASE_CONFIG: dict[str, object] = {
    "cell": "data_driven+hypersphere",
    "n_neurons": 5000,
    "activation": "abs",
    "gain_low": 0.5,
    "gain_high": 2.0,
    "alpha": 1e-3,
    "chunk_size": 5000,
}


register(
    ExperimentSpec(
        name="biases_mnist",
        run_fn=_biases_mnist_run,
        default_run_context=_biases_mnist_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=(
            "B1 panel for H1: bias_mode × encoder_strategy on MNIST. "
            "Knob 'cell' enumerates 9 cells; the headline is the 2×3 panel "
            "of {data_driven, random_normal} × {hypersphere, gaussian, "
            "sparse}.  See analyze_biases_panel.py for the panel-level "
            "claim verdict."
        ),
    )
)
