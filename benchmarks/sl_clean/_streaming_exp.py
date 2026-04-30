"""B2 streaming sMNIST experiment registry.

Imported by :mod:`benchmarks.sl_clean.experiments` so the registration
side-effect is part of the standard registry import.

Knobs:
  * ``n_neurons`` for the H4 scaling sweep
    (``streaming_smnist_row``, ``streaming_smnist_permuted``).
  * ``solve_mode`` ∈ {``batch_acc``, ``accumulate``, ``woodbury``} for
    the H3 equivalence sweep (``streaming_solve_modes``).  All three
    modes share encoders / gains / biases for a given seed so any
    accuracy gap is purely solver numerics.

RNG hygiene: a single ``torch.Generator`` is derived from the seed and
threaded through ``StreamingNEFClassifier`` so encoders, gains and
data-driven biases are bit-identical across solve modes for the same
seed.  No reliance on the global PyTorch RNG.
"""

from __future__ import annotations

import time

import torch

from leenef.streaming import StreamingNEFClassifier

from .datasets import load_dataset, load_smnist_permuted, load_smnist_row
from .sl_repro import set_sl_seed

STREAMING_SOLVE_MODES = ("batch_acc", "accumulate", "woodbury")


def streaming_make_rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + 42)


def _load_streaming_dataset(run_context: dict):
    dataset = str(run_context.get("dataset", "sMNIST-row"))
    if dataset == "sMNIST-row":
        return load_smnist_row()
    if dataset == "sMNIST-permuted":
        perm_seed = int(run_context.get("perm_seed", 0))
        return load_smnist_permuted(perm_seed=perm_seed)
    return load_dataset(
        dataset, preprocessing=str(run_context.get("preprocessing", "div255_flat"))
    )


@torch.no_grad()
def _classification_accuracy(
    clf: StreamingNEFClassifier,
    x_seq: torch.Tensor,
    y: torch.Tensor,
    *,
    eval_batch: int = 2048,
    encode_max_tokens: int = 50_000,
) -> float:
    n_correct = 0
    for s in range(0, x_seq.shape[0], eval_batch):
        e = min(s + eval_batch, x_seq.shape[0])
        A = clf.encode_sequence(x_seq[s:e], max_tokens=encode_max_tokens)
        preds = (A @ clf.decoders).argmax(dim=1)
        n_correct += int((preds == y[s:e]).sum())
    return n_correct / x_seq.shape[0]


def streaming_run(config: dict, seed: int, run_context: dict) -> dict:
    """Run one streaming sMNIST cell."""
    set_sl_seed(seed)
    n_neurons = int(config["n_neurons"])
    window_size = int(config["window_size"])
    encoder_strategy = str(config["encoder_strategy"])
    gain_low = float(config["gain_low"])
    gain_high = float(config["gain_high"])
    alpha = float(config["alpha"])
    alpha_scale = str(config.get("alpha_scale", "trace"))
    if alpha_scale not in ("trace", "fixed"):
        raise ValueError(f"unknown alpha_scale {alpha_scale!r}; expected 'trace' or 'fixed'")
    solve_mode = str(config["solve_mode"])
    batch_size = int(config.get("batch_size", 1000))
    use_centers = bool(config.get("data_driven_biases", True))
    encode_max_tokens = int(config.get("encode_max_tokens", 50_000))
    chunk_stats = bool(config.get("chunk_stats", False))
    eval_train_accuracy = bool(config.get("eval_train_accuracy", False))
    if solve_mode not in STREAMING_SOLVE_MODES:
        raise ValueError(
            f"unknown solve_mode {solve_mode!r}; expected one of {STREAMING_SOLVE_MODES}"
        )

    bundle = _load_streaming_dataset(run_context)
    x_train, y_train = bundle.x_train, bundle.y_train
    x_test, y_test = bundle.x_test, bundle.y_test
    if x_train.dim() != 3:
        raise RuntimeError(
            "streaming experiments require sequential dataset (N, T, d); "
            f"got shape {tuple(x_train.shape)}"
        )
    n_classes = bundle.n_classes
    Y_oh = bundle.one_hot(y_train)

    rng = streaming_make_rng(seed)
    clf = StreamingNEFClassifier(
        d_timestep=int(x_train.shape[2]),
        n_neurons=n_neurons,
        d_out=n_classes,
        window_size=window_size,
        activation=str(config.get("activation", "abs")),
        encoder_strategy=encoder_strategy,
        gain=(gain_low, gain_high),
        rng=rng,
        centers=x_train if use_centers else None,
    )

    t0 = time.perf_counter()
    # When alpha_scale='trace', do a pilot accumulate pass to compute the
    # full-data trace(AᵀA)/n, then set alpha_eff = alpha * trace/n.  This
    # matches solvers.tikhonov / solve_from_normal_equations elsewhere in
    # the codebase and removes the n_neurons-dependent under-regularisation
    # that fixed alpha suffers from at large n.  All three solve modes use
    # the SAME alpha_eff so H3 equivalence stays clean.
    if alpha_scale == "trace":
        clf.accumulate(x_train, Y_oh, max_tokens=encode_max_tokens, chunk_stats=chunk_stats)
        trace_per_neuron = float(torch.diagonal(clf._ata).mean())
        alpha_eff = max(alpha * trace_per_neuron, alpha)
        if solve_mode == "batch_acc":
            clf.solve(alpha=alpha_eff)
        elif solve_mode == "accumulate":
            # Pilot already accumulated everything in one shot — but we
            # want to faithfully exercise the multi-batch accumulate path
            # for H3.  Reset and redo in batches with the locked alpha_eff.
            clf.reset_continuous()
            for s in range(0, x_train.shape[0], batch_size):
                e = min(s + batch_size, x_train.shape[0])
                clf.accumulate(
                    x_train[s:e],
                    Y_oh[s:e],
                    max_tokens=encode_max_tokens,
                    chunk_stats=chunk_stats,
                )
            clf.solve(alpha=alpha_eff)
        else:  # woodbury — needs alpha locked for the M⁻¹ invariant
            clf.reset_continuous()
            for s in range(0, x_train.shape[0], batch_size):
                e = min(s + batch_size, x_train.shape[0])
                clf.continuous_fit(
                    x_train[s:e], Y_oh[s:e], alpha=alpha_eff, max_tokens=encode_max_tokens
                )
            clf.refresh_inverse(alpha=alpha_eff)
    else:  # alpha_scale == 'fixed'
        alpha_eff = alpha
        if solve_mode == "batch_acc":
            clf.accumulate(x_train, Y_oh, max_tokens=encode_max_tokens, chunk_stats=chunk_stats)
            clf.solve(alpha=alpha)
        elif solve_mode == "accumulate":
            for s in range(0, x_train.shape[0], batch_size):
                e = min(s + batch_size, x_train.shape[0])
                clf.accumulate(
                    x_train[s:e],
                    Y_oh[s:e],
                    max_tokens=encode_max_tokens,
                    chunk_stats=chunk_stats,
                )
            clf.solve(alpha=alpha)
        else:  # woodbury
            for s in range(0, x_train.shape[0], batch_size):
                e = min(s + batch_size, x_train.shape[0])
                clf.continuous_fit(
                    x_train[s:e], Y_oh[s:e], alpha=alpha, max_tokens=encode_max_tokens
                )
            clf.refresh_inverse(alpha=alpha)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    train_acc = (
        _classification_accuracy(clf, x_train, y_train, encode_max_tokens=encode_max_tokens)
        if eval_train_accuracy
        else None
    )
    test_acc = _classification_accuracy(clf, x_test, y_test, encode_max_tokens=encode_max_tokens)
    eval_s = time.perf_counter() - t1

    return {
        "test_accuracy": round(test_acc, 6),
        "train_accuracy": None if train_acc is None else round(train_acc, 6),
        "fit_time_s": round(fit_s, 4),
        "eval_time_s": round(eval_s, 4),
        "factors": {
            "solve_mode": solve_mode,
            "n_neurons": n_neurons,
            "window_size": window_size,
            "dataset": bundle.name,
        },
        "diagnostics": {
            "encoder_norm_mean": round(float(clf.encoders.norm(dim=1).mean()), 6),
            "encoder_norm_std": round(float(clf.encoders.norm(dim=1).std()), 6),
            "bias_mean": round(float(clf.bias.mean()), 6),
            "bias_std": round(float(clf.bias.std()), 6),
            "gain_mean": round(float(clf._gain.mean()), 6),
            "alpha_scale": alpha_scale,
            "alpha_nominal": alpha,
            "alpha_eff": round(float(alpha_eff), 9),
        },
        "dataset_metadata": bundle.metadata(),
    }


def streaming_row_context() -> dict:
    return {
        "experiment_kind": "smnist_row_streaming",
        "dataset": "sMNIST-row",
        "preprocessing": "smnist_row_div255",
        "n_train": 60000,
        "n_test": 10000,
        "T": 28,
        "d_timestep": 28,
        "n_classes": 10,
    }


def streaming_permuted_context() -> dict:
    return {
        "experiment_kind": "smnist_permuted_streaming",
        "dataset": "sMNIST-permuted",
        "preprocessing": "smnist_permuted_div255_seed=0",
        "perm_seed": 0,
        "n_train": 60000,
        "n_test": 10000,
        "T": 784,
        "d_timestep": 1,
        "n_classes": 10,
    }


STREAMING_ROW_BASE_CONFIG: dict[str, object] = {
    "n_neurons": 4000,
    "window_size": 10,
    "encoder_strategy": "hypersphere",
    "activation": "abs",
    "gain_low": 0.5,
    "gain_high": 2.0,
    "alpha": 1e-2,
    "alpha_scale": "trace",
    "solve_mode": "batch_acc",
    "batch_size": 1000,
    "data_driven_biases": True,
    "encode_max_tokens": 50_000,
    "chunk_stats": False,
    "eval_train_accuracy": False,
}

STREAMING_PERMUTED_BASE_CONFIG: dict[str, object] = {
    **STREAMING_ROW_BASE_CONFIG,
    "n_neurons": 4000,
    "window_size": 10,
}

STREAMING_DESCRIPTION_ROW = (
    "B2 streaming sMNIST-row: delay-line StreamingNEFClassifier with "
    "data-driven biases.  Headline knob is n_neurons; H4 keystone."
)
STREAMING_DESCRIPTION_PERMUTED = (
    "B2 streaming sMNIST-permuted: delay-line StreamingNEFClassifier "
    "on permuted-pixel MNIST (T=784, d=1).  Companion to the row "
    "panel; permutation seed is fixed in run_context."
)
STREAMING_DESCRIPTION_MODES = (
    "B2 streaming solve-mode equivalence (H3): batch_acc vs accumulate "
    "vs woodbury at fixed n_neurons.  Encoders/gains/biases are shared "
    "across modes for a given seed so any gap is solver numerics."
)
