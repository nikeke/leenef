"""C2 — activation × gain ablation.

Knob is ``cell`` = ``"{activation}+g{gain_low}_{gain_high}"``, e.g.
``"abs+g0.5_2.0"``.  15-cell grid:
  activation ∈ {abs, relu, tanh, step, square}
  gain      ∈ {(0.5,2.0), (1.0,1.0), (0.1,5.0)}
where (1.0,1.0) is the degenerate fixed-gain control and (0.1,5.0) is
a wider gain spread to test the diversity hypothesis.

Expected outcome: abs ≥ relu (because abs is two-sided so the same
neuron count covers twice the input space) > tanh (saturating costs
a few pp); U(0.5,2.0) wins over fixed gain (per-neuron diversity);
U(0.1,5.0) likely matches U(0.5,2.0) within seed noise (the precise
range matters less than having any spread).

Encoder fixed to ``hypersphere`` and ``data_driven`` biases (project
default).  ``n_neurons=5000``.  10 seeds.
"""

from __future__ import annotations

from ._phase_c_shared import (
    _make_subrngs_phase_c,
    build_encoders_and_biases,
    chunked_accumulate_solve,
    chunked_classification_accuracy,
    load_dataset_for_phase_c,
    set_sl_seed,
)

_C2_ACTIVATIONS = (
    "abs",
    "relu",
    "tanh",
    "step",
    "square",
    "leaky_relu",
    "softplus",
    "bent_line",
    "double_fold",
    "sqrt_abs",
    "cos",
    "gauss",
    "cos_sin",
    "square_wave",
    "square_floor",
    "box",
    "staircase",
    "gelu",
    "silu",
)
_C2_GAIN_RANGES: tuple[tuple[float, float], ...] = (
    (0.5, 2.0),  # project default
    (1.0, 1.0),  # fixed-gain control
    (0.1, 5.0),  # wider spread
)


def _gain_token(low: float, high: float) -> str:
    return f"g{low}_{high}"


C2_CELLS: tuple[str, ...] = tuple(
    f"{act}+{_gain_token(low, high)}" for act in _C2_ACTIVATIONS for (low, high) in _C2_GAIN_RANGES
)


def _parse_cell(cell: str) -> tuple[str, float, float]:
    if "+" not in cell:
        raise ValueError(f"C2 cell {cell!r}: expected 'act+g{{low}}_{{high}}'")
    act, gain = cell.split("+", 1)
    if act not in _C2_ACTIVATIONS:
        raise ValueError(f"C2 cell {cell!r}: unknown activation {act!r}")
    if not gain.startswith("g") or "_" not in gain:
        raise ValueError(f"C2 cell {cell!r}: gain token must look like 'g{{low}}_{{high}}'")
    low_s, high_s = gain[1:].split("_", 1)
    return act, float(low_s), float(high_s)


def _c2_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    activation, gain_low, gain_high = _parse_cell(str(config["cell"]))
    n_neurons = int(config["n_neurons"])
    alpha = float(config["alpha"])
    chunk_size = int(config.get("chunk_size", 5000))
    encoder_strategy = str(config.get("encoder_strategy", "hypersphere"))

    bundle = load_dataset_for_phase_c(run_context)
    x_train, y_train = bundle.x_train, bundle.y_train
    x_test, y_test = bundle.x_test, bundle.y_test
    d_in = int(x_train.shape[1])
    image_shape = run_context.get("image_shape")
    if image_shape is not None:
        image_shape = tuple(image_shape)  # type: ignore[assignment]

    rngs = _make_subrngs_phase_c(seed)
    enc, gain, bias = build_encoders_and_biases(
        strategy=encoder_strategy,
        n_neurons=n_neurons,
        d_in=d_in,
        rng_encoder=rngs["encoder"],
        rng_gain=rngs["gain"],
        rng_center=rngs["center_index"],
        x_train=x_train,
        y_train=y_train,
        image_shape=image_shape,  # type: ignore[arg-type]
        patch_size=int(config.get("patch_size", 5)),
        gain_low=gain_low,
        gain_high=gain_high,
    )

    Y_oh = bundle.one_hot(y_train).float()
    D, timings = chunked_accumulate_solve(
        enc=enc,
        gain=gain,
        bias=bias,
        activation=activation,
        x_train=x_train,
        y_train_targets=Y_oh,
        chunk_size=chunk_size,
        alpha=(alpha, "trace"),
    )

    train_acc = chunked_classification_accuracy(
        enc=enc,
        gain=gain,
        bias=bias,
        activation=activation,
        D=D,
        x=x_train,
        y=y_train,
        chunk_size=chunk_size,
    )
    test_acc = chunked_classification_accuracy(
        enc=enc,
        gain=gain,
        bias=bias,
        activation=activation,
        D=D,
        x=x_test,
        y=y_test,
        chunk_size=chunk_size,
    )

    return {
        "test_accuracy": round(test_acc, 6),
        "train_accuracy": round(train_acc, 6),
        **timings,
        "factors": {
            "activation": activation,
            "gain_low": gain_low,
            "gain_high": gain_high,
        },
        "diagnostics": {
            "encoder_norm_mean": round(float(enc.norm(dim=1).mean()), 6),
            "bias_mean": round(float(bias.mean()), 6),
            "bias_std": round(float(bias.std()), 6),
            "gain_mean": round(float(gain.mean()), 6),
            "gain_std": round(float(gain.std()), 6),
        },
        "dataset_metadata": bundle.metadata(),
    }


def _c2_mnist_context() -> dict:
    return {
        "experiment_kind": "mnist_singlelayer_c2_activation_panel",
        "dataset": "MNIST",
        "preprocessing": "div255_flat",
        "n_train": 60000,
        "n_test": 10000,
        "d_in": 784,
        "n_classes": 10,
        "image_shape": [28, 28],
    }


def _c2_fashion_context() -> dict:
    return {
        "experiment_kind": "fashion_singlelayer_c2_activation_panel",
        "dataset": "FashionMNIST",
        "preprocessing": "div255_flat",
        "n_train": 60000,
        "n_test": 10000,
        "d_in": 784,
        "n_classes": 10,
        "image_shape": [28, 28],
    }


def _c2_cifar10_context() -> dict:
    return {
        "experiment_kind": "cifar10_singlelayer_c2_activation_panel",
        "dataset": "CIFAR10",
        "preprocessing": "div255_flat",
        "n_train": 50000,
        "n_test": 10000,
        "d_in": 3072,
        "n_classes": 10,
        "image_shape": [32, 32],
    }


C2_BASE_CONFIG: dict[str, object] = {
    "cell": "abs+g0.5_2.0",
    "n_neurons": 5000,
    "alpha": 1e-3,
    "chunk_size": 5000,
    "encoder_strategy": "hypersphere",
    "patch_size": 5,
}

C2_CIFAR10_BASE_CONFIG: dict[str, object] = {**C2_BASE_CONFIG, "chunk_size": 2500}


C2_DESCRIPTION = (
    "C2 activation × gain-range ablation. Knob 'cell' = "
    "'{activation}+g{low}_{high}'.  15-cell grid: "
    "activation ∈ {abs, relu, tanh, step, square}, "
    "gain ∈ {(0.5,2.0), (1.0,1.0), (0.1,5.0)}.  "
    "n_neurons=5000, encoder=hypersphere, bias=data_driven, alpha=1e-3."
)
