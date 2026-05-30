"""Phase-C shared helpers — chunked accumulate+solve for single-layer NEF.

C1 (scaling), C2 (activation × gain), C7 (housing regression) all reduce
to: build encoders + per-neuron gains + data-driven biases, accumulate
``AᵀA`` and ``AᵀY`` chunk by chunk, solve once with Cholesky.

The shared core is parameterised by ``activation``, ``gain_low``,
``gain_high``, ``encoder_strategy`` and ``n_neurons``.  Sub-RNGs are
derived deterministically from the seed via :func:`_make_subrngs` so
that varying any one factor produces bit-identical encoders / gains
/ centers under all the other factors.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from leenef.encoders import make_encoders
from leenef.solvers import AlphaPolicy, solve_from_normal_equations

from .datasets import load_dataset
from .sl_repro import set_sl_seed


def _make_subrngs_phase_c(seed: int) -> dict[str, torch.Generator]:
    """Phase-C sub-RNG streams.

    We re-derive these locally so that adding new streams (e.g.
    ``ensemble_member`` for C3) does not perturb the seeds used by the
    B-phase experiments.  The integer literals are independent of the
    ones in ``experiments._make_subrngs``.
    """
    streams = ("encoder", "gain", "center_index", "ensemble_member", "init")
    return {
        name: torch.Generator().manual_seed(seed * 2_000_003 + 31 * i + 7)
        for i, name in enumerate(streams)
    }


def _activation_fn(name: str) -> Callable[[torch.Tensor], None]:
    """Return an in-place activation matching ``leenef.activations``."""
    if name == "abs":
        return torch.Tensor.abs_
    if name == "relu":
        return lambda t: t.clamp_(min=0.0)
    if name == "leaky_relu":
        return lambda t: torch.nn.functional.leaky_relu_(t, 0.1)
    if name == "softplus":
        return lambda t: t.copy_(torch.nn.functional.softplus(t))
    if name == "tanh":
        return torch.Tensor.tanh_
    if name == "step":
        return lambda t: t.copy_(t > 0.0)
    if name == "square":
        return torch.Tensor.square_
    if name == "bent_line":
        # f(z) = z for z>=0 else 0.5 z  ==  0.75 z + 0.25 |z|
        return lambda t: t.copy_(torch.where(t >= 0, t, 0.5 * t))
    if name == "double_fold":
        # f(z) = ||z| - 1|
        return lambda t: t.abs_().sub_(1.0).abs_()
    if name == "sqrt_abs":
        return lambda t: t.abs_().sqrt_()
    if name == "cos":
        return torch.Tensor.cos_
    if name == "gauss":
        # f(z) = exp(-z^2 / 2)
        return lambda t: t.square_().mul_(-0.5).exp_()
    if name == "cos_sin":
        # f(z) = cos(z) + sin(z)
        return lambda t: t.copy_(torch.cos(t) + torch.sin(t))
    if name == "square_wave":
        # f(z) = sign(cos(z))
        return lambda t: t.copy_(torch.sign(torch.cos(t)))
    if name == "box":
        # f(z) = 1[|z| < 1]
        return lambda t: t.copy_((t.abs() < 1.0).to(t.dtype))
    if name == "gelu":
        return lambda t: t.copy_(torch.nn.functional.gelu(t))
    if name == "silu":
        return lambda t: t.copy_(torch.nn.functional.silu(t))
    raise ValueError(
        f"unknown activation {name!r}; phase-c supports "
        "abs/relu/leaky_relu/softplus/tanh/step/square/bent_line/"
        "double_fold/sqrt_abs/cos/gauss/cos_sin/square_wave/box/gelu/silu"
    )


def build_encoders_and_biases(
    *,
    strategy: str,
    n_neurons: int,
    d_in: int,
    rng_encoder: torch.Generator,
    rng_gain: torch.Generator,
    rng_center: torch.Generator,
    x_train: torch.Tensor,
    y_train: torch.Tensor | None,
    image_shape: tuple[int, int] | None,
    patch_size: int,
    gain_low: float,
    gain_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(encoders, gain, bias)`` ready for a chunked accumulate."""
    kwargs: dict[str, object] = {"strategy": strategy, "rng": rng_encoder}
    if strategy == "receptive_field":
        if image_shape is None:
            raise ValueError("receptive_field needs image_shape")
        kwargs["image_shape"] = image_shape
        kwargs["patch_size"] = patch_size
    elif strategy in ("whitened", "local_pca"):
        kwargs["train_data"] = x_train
    elif strategy == "class_contrast":
        if y_train is None:
            raise ValueError("class_contrast needs y_train")
        kwargs["train_data"] = x_train
        kwargs["train_labels"] = y_train

    out = make_encoders(n_neurons, d_in, **kwargs)
    if isinstance(out, tuple):
        enc, centers = out
    else:
        enc = out
        idx = torch.randint(len(x_train), (n_neurons,), generator=rng_center)
        centers = x_train[idx]

    gain = gain_low + (gain_high - gain_low) * torch.rand(n_neurons, generator=rng_gain)
    bias = -gain * (centers * enc).sum(dim=1)
    return enc, gain, bias


def chunked_accumulate_solve(
    *,
    enc: torch.Tensor,
    gain: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
    x_train: torch.Tensor,
    y_train_targets: torch.Tensor,
    chunk_size: int,
    alpha: AlphaPolicy,
) -> tuple[torch.Tensor, dict]:
    """Run the accumulate + Cholesky solve and return ``(D, timings)``.

    ``y_train_targets`` is whatever target tensor the experiment wants
    decoded — one-hot for classification, regression target for
    regression.
    """
    n_neurons = enc.shape[0]
    d_out = y_train_targets.shape[1]
    act = _activation_fn(activation)

    ATA = torch.zeros(n_neurons, n_neurons, dtype=torch.float32)
    ATY = torch.zeros(n_neurons, d_out, dtype=torch.float32)

    t0 = time.perf_counter()
    for s in range(0, x_train.shape[0], chunk_size):
        e = min(s + chunk_size, x_train.shape[0])
        z = x_train[s:e] @ enc.T
        z.mul_(gain)
        z.add_(bias)
        act(z)
        ATA.addmm_(z.T, z)
        ATY.addmm_(z.T, y_train_targets[s:e])
    encode_fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    D = solve_from_normal_equations(ATA, ATY, alpha=alpha)
    solve_s = time.perf_counter() - t0

    return D, {"encode_fit_time_s": round(encode_fit_s, 4), "solve_time_s": round(solve_s, 4)}


def chunked_classification_accuracy(
    *,
    enc: torch.Tensor,
    gain: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
    D: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int,
) -> float:
    act = _activation_fn(activation)
    n_correct = 0
    for s in range(0, x.shape[0], chunk_size):
        e = min(s + chunk_size, x.shape[0])
        z = x[s:e] @ enc.T
        z.mul_(gain)
        z.add_(bias)
        act(z)
        preds = (z @ D).argmax(dim=1)
        n_correct += int((preds == y[s:e]).sum())
    return n_correct / x.shape[0]


def chunked_regression_mse(
    *,
    enc: torch.Tensor,
    gain: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
    D: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int,
) -> float:
    act = _activation_fn(activation)
    sse = 0.0
    n = 0
    for s in range(0, x.shape[0], chunk_size):
        e = min(s + chunk_size, x.shape[0])
        z = x[s:e] @ enc.T
        z.mul_(gain)
        z.add_(bias)
        act(z)
        preds = z @ D
        diff = preds - y[s:e]
        sse += float((diff * diff).sum())
        n += int(diff.numel())
    return sse / max(n, 1)


def load_dataset_for_phase_c(run_context: dict):
    return load_dataset(
        run_context["dataset"], preprocessing=str(run_context.get("preprocessing", "div255_flat"))
    )


__all__ = [
    "build_encoders_and_biases",
    "chunked_accumulate_solve",
    "chunked_classification_accuracy",
    "chunked_regression_mse",
    "load_dataset_for_phase_c",
    "set_sl_seed",
    "_make_subrngs_phase_c",
]
