"""C1 — single-layer neuron-count scaling experiment.

Knob is ``cell``, encoding ``n_neurons`` directly (e.g. ``"n2000"``).
Default encoder is ``hypersphere`` with data-driven biases (project
default).  Activation, gain range, alpha are taken from base config.

Expected outcome (TR §5.1): monotonic test-accuracy improvement with
``n_neurons``, plateauing somewhere in the 5k–20k range.  Used to
populate the scaling figure F-scaling and to confirm the n=5000
default chosen elsewhere is in the plateau region.
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

C1_CELLS: tuple[str, ...] = ("n500", "n1000", "n2000", "n5000", "n10000", "n20000")


def _parse_cell(cell: str) -> int:
    if not cell.startswith("n"):
        raise ValueError(f"C1 cell {cell!r}: expected 'n{{n_neurons}}'")
    return int(cell[1:])


def _c1_run(config: dict, seed: int, run_context: dict) -> dict:
    set_sl_seed(seed)
    n_neurons = _parse_cell(str(config["cell"]))
    alpha = float(config["alpha"])
    gain_low = float(config["gain_low"])
    gain_high = float(config["gain_high"])
    chunk_size = int(config.get("chunk_size", 5000))
    activation = str(config.get("activation", "abs"))
    encoder_strategy = str(config.get("encoder_strategy", "hypersphere"))

    bundle = load_dataset_for_phase_c(run_context)
    x_train, y_train = bundle.x_train, bundle.y_train
    x_test, y_test = bundle.x_test, bundle.y_test
    d_in = int(x_train.shape[1])
    n_classes = bundle.n_classes
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
        "factors": {"n_neurons": n_neurons, "encoder_strategy": encoder_strategy},
        "diagnostics": {
            "n_neurons": n_neurons,
            "encoder_norm_mean": round(float(enc.norm(dim=1).mean()), 6),
            "bias_mean": round(float(bias.mean()), 6),
            "bias_std": round(float(bias.std()), 6),
            "gain_mean": round(float(gain.mean()), 6),
            "n_classes": int(n_classes),
        },
        "dataset_metadata": bundle.metadata(),
    }


def _c1_make_context(
    *, dataset: str, n_train: int, n_test: int, d_in: int, image_shape: tuple[int, int]
):
    def _ctx() -> dict:
        return {
            "experiment_kind": f"{dataset.lower()}_singlelayer_c1_scaling_panel",
            "dataset": dataset,
            "preprocessing": "div255_flat",
            "n_train": n_train,
            "n_test": n_test,
            "d_in": d_in,
            "n_classes": 10,
            "image_shape": list(image_shape),
        }

    return _ctx


def _c1_mnist_context() -> dict:
    return _c1_make_context(
        dataset="MNIST", n_train=60000, n_test=10000, d_in=784, image_shape=(28, 28)
    )()


def _c1_fashion_context() -> dict:
    return _c1_make_context(
        dataset="FashionMNIST", n_train=60000, n_test=10000, d_in=784, image_shape=(28, 28)
    )()


def _c1_cifar10_context() -> dict:
    return _c1_make_context(
        dataset="CIFAR10", n_train=50000, n_test=10000, d_in=3072, image_shape=(32, 32)
    )()


C1_BASE_CONFIG: dict[str, object] = {
    "cell": "n5000",
    "alpha": 1e-3,
    "gain_low": 0.5,
    "gain_high": 2.0,
    "chunk_size": 5000,
    "activation": "abs",
    "encoder_strategy": "hypersphere",
    "patch_size": 5,
}

C1_CIFAR10_BASE_CONFIG: dict[str, object] = {**C1_BASE_CONFIG, "chunk_size": 2500}


C1_DESCRIPTION = (
    "C1 single-layer neuron-count scaling. Knob 'cell' = 'n{n_neurons}'. "
    "Cells: " + ",".join(C1_CELLS) + ". "
    "Encoder=hypersphere, bias=data_driven, activation=abs, "
    "gain=U(0.5,2.0), alpha=1e-3.  Reports test_accuracy mean/SE per n."
)
