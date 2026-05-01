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

from .datasets import load_dataset
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


def _biases_run(config: dict, seed: int, run_context: dict) -> dict:
    """Generic data-driven-biases panel runner.

    The dataset is read from ``run_context['dataset']`` (one of
    ``MNIST``, ``FashionMNIST``, ``CIFAR10``).  Everything else is
    dataset-agnostic: encoders, gain, bias_mode, chunked Cholesky solve,
    accuracy on the test split.

    The cache key already includes the full ``run_context`` so two
    different datasets cannot collide.
    """
    set_sl_seed(seed)
    bias_mode, encoder_strategy = _parse_biases_cell(str(config["cell"]))
    n_neurons = int(config["n_neurons"])
    alpha = float(config["alpha"])
    gain_low = float(config["gain_low"])
    gain_high = float(config["gain_high"])
    chunk_size = int(config.get("chunk_size", 5000))

    dataset_name = str(run_context.get("dataset", "MNIST"))
    bundle = load_dataset(
        dataset_name,
        preprocessing=str(run_context.get("preprocessing", "div255_flat")),
    )
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


# Back-compat alias.  Tests still import the old name.
_biases_mnist_run = _biases_run


def _make_biases_context(
    *, dataset: str, n_train: int, n_test: int, d_in: int
) -> Callable[[], dict]:
    def _ctx() -> dict:
        return {
            "experiment_kind": f"{dataset.lower()}_singlelayer_biases_panel",
            "dataset": dataset,
            "preprocessing": "div255_flat",
            "n_train": n_train,
            "n_test": n_test,
            "d_in": d_in,
            "n_classes": 10,
        }

    return _ctx


def _biases_mnist_context() -> dict:
    return _make_biases_context(dataset="MNIST", n_train=60000, n_test=10000, d_in=784)()


def _biases_fashion_context() -> dict:
    return _make_biases_context(dataset="FashionMNIST", n_train=60000, n_test=10000, d_in=784)()


def _biases_cifar10_context() -> dict:
    return _make_biases_context(dataset="CIFAR10", n_train=50000, n_test=10000, d_in=3072)()


def _biases_cifar10_centered_context() -> dict:
    ctx = _make_biases_context(dataset="CIFAR10", n_train=50000, n_test=10000, d_in=3072)()
    ctx["preprocessing"] = "div255_flat_per_channel_centered"
    ctx["experiment_kind"] = "cifar10_centered_singlelayer_biases_panel"
    return ctx


def _biases_mnist_decentered_context() -> dict:
    ctx = _make_biases_context(dataset="MNIST", n_train=60000, n_test=10000, d_in=784)()
    ctx["preprocessing"] = "div255_flat_dc_shift"
    ctx["experiment_kind"] = "mnist_decentered_singlelayer_biases_panel"
    return ctx


BIASES_MNIST_BASE_CONFIG: dict[str, object] = {
    "cell": "data_driven+hypersphere",
    "n_neurons": 5000,
    "activation": "abs",
    "gain_low": 0.5,
    "gain_high": 2.0,
    "alpha": 1e-3,
    "chunk_size": 5000,
}


# Default base configs for the Fashion-MNIST and CIFAR-10 panels.
# Same shape as MNIST; CIFAR-10 lowers chunk_size to bound peak memory
# of the (chunk × n_neurons) activity matrix on a 12-thread CPU.
BIASES_FASHION_BASE_CONFIG: dict[str, object] = dict(BIASES_MNIST_BASE_CONFIG)
BIASES_CIFAR10_BASE_CONFIG: dict[str, object] = {
    **BIASES_MNIST_BASE_CONFIG,
    "chunk_size": 2500,
}


_BIASES_DESCRIPTION = (
    "B1 panel for H1: bias_mode × encoder_strategy. Knob 'cell' "
    "enumerates 9 cells; the headline is the 2×3 panel of "
    "{data_driven, random_normal} × {hypersphere, gaussian, sparse}. "
    "Dataset is taken from run_context. See analyze_biases_panel.py "
    "for the panel-level claim verdict."
)

register(
    ExperimentSpec(
        name="biases_mnist",
        run_fn=_biases_run,
        default_run_context=_biases_mnist_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_BIASES_DESCRIPTION + " (MNIST)",
    )
)

register(
    ExperimentSpec(
        name="biases_fashion",
        run_fn=_biases_run,
        default_run_context=_biases_fashion_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_BIASES_DESCRIPTION + " (Fashion-MNIST)",
    )
)

register(
    ExperimentSpec(
        name="biases_cifar10",
        run_fn=_biases_run,
        default_run_context=_biases_cifar10_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_BIASES_DESCRIPTION + " (CIFAR-10)",
    )
)

register(
    ExperimentSpec(
        name="biases_cifar10_centered",
        run_fn=_biases_run,
        default_run_context=_biases_cifar10_centered_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_BIASES_DESCRIPTION
        + " (CIFAR-10 per-channel centred — DC-fold falsification)",
    )
)

register(
    ExperimentSpec(
        name="biases_mnist_decentered",
        run_fn=_biases_run,
        default_run_context=_biases_mnist_decentered_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_BIASES_DESCRIPTION + " (MNIST DC-shifted — DC-fold falsification)",
    )
)


# ---------------------------------------------------------------------------
# B3 — encoder-strategy ablation.
#
# Knob is ``encoder_strategy`` (the cell value), enumerating all 7
# strategies registered in ``leenef.encoders``.  Bias mode is fixed to
# ``data_driven`` (the project default).  Activation, gain range,
# n_neurons, alpha, chunk_size are taken from the base config.
#
# Strategy-specific kwargs are wired here:
#   - ``hypersphere`` / ``gaussian`` / ``sparse``: only ``rng``.
#   - ``receptive_field``: needs ``image_shape`` from run_context.
#     Default ``patch_size = 5``.
#   - ``whitened``, ``class_contrast``, ``local_pca``: need
#     ``train_data`` (and ``train_labels`` for class_contrast).
#   - ``class_contrast`` / ``local_pca`` return ``(encoders, centers)``;
#     we use those returned centers for the data-driven biases instead
#     of fresh random samples (they are the natural reference points
#     for those strategies).
# ---------------------------------------------------------------------------


_B3_STRATEGIES = (
    "hypersphere",
    "gaussian",
    "sparse",
    "receptive_field",
    "whitened",
    "class_contrast",
    "local_pca",
)


def _b3_make_encoders_and_centers(
    *,
    strategy: str,
    n_neurons: int,
    d_in: int,
    rng: torch.Generator,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    image_shape: tuple[int, int] | None,
    patch_size: int,
    center_index_rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build encoders and per-neuron centers for a given strategy.

    Centers are training samples used to derive data-driven biases via
    ``b = -gain * (center · encoder)``.  For ``class_contrast`` and
    ``local_pca`` we honor the centers returned by the encoder builder;
    otherwise we sample uniformly with ``center_index_rng``.
    """
    kwargs: dict[str, object] = {"strategy": strategy, "rng": rng}
    if strategy == "receptive_field":
        if image_shape is None:
            raise ValueError("receptive_field needs image_shape in run_context")
        kwargs["image_shape"] = image_shape
        kwargs["patch_size"] = patch_size
    elif strategy in ("whitened", "local_pca"):
        kwargs["train_data"] = x_train
    elif strategy == "class_contrast":
        kwargs["train_data"] = x_train
        kwargs["train_labels"] = y_train

    out = make_encoders(n_neurons, d_in, **kwargs)
    if isinstance(out, tuple):
        enc, centers = out
    else:
        enc = out
        idx = torch.randint(len(x_train), (n_neurons,), generator=center_index_rng)
        centers = x_train[idx]
    return enc, centers


def _b3_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    strategy = str(config["cell"])
    if strategy not in _B3_STRATEGIES:
        raise ValueError(f"unknown encoder strategy {strategy!r}; have {_B3_STRATEGIES}")
    n_neurons = int(config["n_neurons"])
    alpha = float(config["alpha"])
    gain_low = float(config["gain_low"])
    gain_high = float(config["gain_high"])
    chunk_size = int(config.get("chunk_size", 5000))
    patch_size = int(config.get("patch_size", 5))

    dataset_name = str(run_context.get("dataset", "MNIST"))
    bundle = load_dataset(
        dataset_name,
        preprocessing=str(run_context.get("preprocessing", "div255_flat")),
    )
    x_train, y_train = bundle.x_train, bundle.y_train
    x_test, y_test = bundle.x_test, bundle.y_test
    d_in = int(x_train.shape[1])
    n_classes = bundle.n_classes
    image_shape = run_context.get("image_shape")
    if image_shape is not None:
        image_shape = tuple(image_shape)  # type: ignore[assignment]

    rngs = _make_subrngs(seed)
    gain = gain_low + (gain_high - gain_low) * torch.rand(n_neurons, generator=rngs["gain"])

    enc, centers = _b3_make_encoders_and_centers(
        strategy=strategy,
        n_neurons=n_neurons,
        d_in=d_in,
        rng=rngs["encoder"],
        x_train=x_train,
        y_train=y_train,
        image_shape=image_shape,  # type: ignore[arg-type]
        patch_size=patch_size,
        center_index_rng=rngs["center_index"],
    )
    bias = -gain * (centers * enc).sum(dim=1)

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
        "factors": {"encoder_strategy": strategy, "bias_mode": "data_driven"},
        "diagnostics": {
            "encoder_norm_mean": round(float(enc_norms.mean()), 6),
            "encoder_norm_std": round(float(enc_norms.std()), 6),
            "bias_mean": round(float(bias.mean()), 6),
            "bias_std": round(float(bias.std()), 6),
            "gain_mean": round(float(gain.mean()), 6),
        },
        "dataset_metadata": bundle.metadata(),
    }


def _b3_make_context(
    *, dataset: str, n_train: int, n_test: int, d_in: int, image_shape: tuple[int, int]
) -> Callable[[], dict]:
    def _ctx() -> dict:
        return {
            "experiment_kind": f"{dataset.lower()}_singlelayer_b3_strategies_panel",
            "dataset": dataset,
            "preprocessing": "div255_flat",
            "n_train": n_train,
            "n_test": n_test,
            "d_in": d_in,
            "n_classes": 10,
            "image_shape": list(image_shape),
        }

    return _ctx


def _b3_mnist_context() -> dict:
    return _b3_make_context(
        dataset="MNIST", n_train=60000, n_test=10000, d_in=784, image_shape=(28, 28)
    )()


def _b3_fashion_context() -> dict:
    return _b3_make_context(
        dataset="FashionMNIST", n_train=60000, n_test=10000, d_in=784, image_shape=(28, 28)
    )()


def _b3_cifar10_context() -> dict:
    return _b3_make_context(
        dataset="CIFAR10", n_train=50000, n_test=10000, d_in=3072, image_shape=(32, 32)
    )()


B3_BASE_CONFIG: dict[str, object] = {
    "cell": "hypersphere",
    "n_neurons": 5000,
    "activation": "abs",
    "gain_low": 0.5,
    "gain_high": 2.0,
    "alpha": 1e-3,
    "chunk_size": 5000,
    "patch_size": 5,
}

B3_CIFAR10_BASE_CONFIG: dict[str, object] = {**B3_BASE_CONFIG, "chunk_size": 2500}


_B3_DESCRIPTION = (
    "B3 panel: encoder strategy ablation. Knob 'cell' enumerates the 7 "
    "registered strategies (hypersphere, gaussian, sparse, "
    "receptive_field, whitened, class_contrast, local_pca). "
    "bias_mode fixed to data_driven (project default per H7). "
    "n_neurons fixed via base config. Per-cell mean/SE reported."
)


register(
    ExperimentSpec(
        name="b3_strategies_mnist",
        run_fn=_b3_run,
        default_run_context=_b3_mnist_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_B3_DESCRIPTION + " (MNIST)",
    )
)

register(
    ExperimentSpec(
        name="b3_strategies_fashion",
        run_fn=_b3_run,
        default_run_context=_b3_fashion_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_B3_DESCRIPTION + " (Fashion-MNIST)",
    )
)

register(
    ExperimentSpec(
        name="b3_strategies_cifar10",
        run_fn=_b3_run,
        default_run_context=_b3_cifar10_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_B3_DESCRIPTION + " (CIFAR-10)",
    )
)


# ---------------------------------------------------------------------------
# B2 — streaming sMNIST.  Implementation lives in _streaming_exp.py to keep
# this file from sprawling further.  We register the three experiments here.
# ---------------------------------------------------------------------------

from . import _streaming_exp as _streaming  # noqa: E402

register(
    ExperimentSpec(
        name="streaming_smnist_row",
        run_fn=_streaming.streaming_run,
        default_run_context=_streaming.streaming_row_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_streaming.STREAMING_DESCRIPTION_ROW,
    )
)

register(
    ExperimentSpec(
        name="streaming_smnist_permuted",
        run_fn=_streaming.streaming_run,
        default_run_context=_streaming.streaming_permuted_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_streaming.STREAMING_DESCRIPTION_PERMUTED,
    )
)

register(
    ExperimentSpec(
        name="streaming_solve_modes",
        run_fn=_streaming.streaming_run,
        default_run_context=_streaming.streaming_row_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_streaming.STREAMING_DESCRIPTION_MODES,
    )
)

register(
    ExperimentSpec(
        name="streaming_alpha_sweep",
        run_fn=_streaming.streaming_run,
        default_run_context=_streaming.streaming_row_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=(
            "B2 alpha calibration: cell='alpha_scale+alpha+n_neurons'. "
            "Diagnoses the trace-vs-fixed regularisation choice for streaming."
        ),
    )
)


# ---------------------------------------------------------------------------
# B4 — MLP baselines (H2 honest comparison).  Implementation lives in
# _b4_mlp_exp.py.  Knob is ``cell`` over a 12-config grid of
# (depth, width, lr).
# ---------------------------------------------------------------------------

from . import _b4_mlp_exp as _b4  # noqa: E402

register(
    ExperimentSpec(
        name="b4_mlp_mnist",
        run_fn=_b4._b4_run,
        default_run_context=_b4._b4_mnist_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_b4.B4_DESCRIPTION + " (MNIST)",
    )
)

register(
    ExperimentSpec(
        name="b4_mlp_fashion",
        run_fn=_b4._b4_run,
        default_run_context=_b4._b4_fashion_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_b4.B4_DESCRIPTION + " (Fashion-MNIST)",
    )
)

register(
    ExperimentSpec(
        name="b4_mlp_cifar10",
        run_fn=_b4._b4_run,
        default_run_context=_b4._b4_cifar10_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_b4.B4_DESCRIPTION + " (CIFAR-10)",
    )
)

# ---------------------------------------------------------------------------
# B5 — LSTM baseline (H4/H5).
# ---------------------------------------------------------------------------

from . import _b5_lstm_exp as _b5  # noqa: E402

register(
    ExperimentSpec(
        name="b5_lstm_smnist_row",
        run_fn=_b5._b5_run,
        default_run_context=_b5._b5_smnist_row_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_b5.B5_DESCRIPTION + " (sMNIST-row)",
    )
)

register(
    ExperimentSpec(
        name="b5_lstm_smnist_permuted",
        run_fn=_b5._b5_run,
        default_run_context=_b5._b5_smnist_permuted_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=_b5.B5_DESCRIPTION + " (sMNIST-permuted-seed0)",
    )
)
