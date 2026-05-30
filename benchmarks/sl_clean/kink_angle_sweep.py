"""Kink-angle sweep — does the angle at the origin matter?

A single-kink ("bent line") activation with right-arm slope fixed to 1
(45 deg) and interior angle ``theta`` between the two arms has left-arm
slope ``b = tan(pi/4 + theta)``.  In the geometric convention used here:

    theta = pi/2   -> abs        (left slope -1)
    theta = 3pi/4  -> ReLU       (left slope  0)
    theta = pi     -> identity   (left slope +1)  == LINEAR COLLAPSE
    theta in (pi/4, 5pi/4)       -> the left arm is a valid left-pointing ray.

Decompose every such activation as ``G_theta(t) = c1 t + c2 |t|`` with
``c1 = (1 + b)/2`` and ``c2 = (1 - b)/2``.  Because ridge with trace-scaled
alpha is invariant to a global feature rescale, we use the *normalized*
direction ``(c1, c2)/||(c1, c2)||`` so the family stays bounded across the
whole open angle range (the raw slope ``b`` diverges at the endpoints).

The crucial efficiency (and conceptual) point: for a fixed seed every angle
reuses the **same** signed pre-activation ``Z`` and its magnitude
``A = |Z|``.  So we accumulate the five Gram / target blocks

    ZtZ = Z^T Z,  ZtA = Z^T A,  AtA = A^T A,  ZtY = Z^T Y,  AtY = A^T Y

*once* per seed, then each angle is a cheap recombination

    (M^T M) = c1^2 ZtZ + c1 c2 (ZtA + ZtA^T) + c2^2 AtA
    (M^T Y) = c1 ZtY + c2 AtY

followed by a trace-scaled solve.  Test predictions reuse a cached signed
test pre-activation ``Z_test``.

Theory prediction (docs/activation_kernel_theory.md): test accuracy is flat
across the whole sweep except a sharp dip at ``theta = pi`` (the linear
angle), because every non-linear angle shares the identical ``|z|`` features
and only the scalar mix changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from leenef.solvers import solve_from_normal_equations

from ._c2_activation_exp import (
    _c2_cifar10_context,
    _c2_fashion_context,
    _c2_mnist_context,
)
from ._phase_c_shared import (
    _make_subrngs_phase_c,
    build_encoders_and_biases,
    load_dataset_for_phase_c,
    set_sl_seed,
)
from .sl_repro import RESULTS_DIR, code_identity, hardware_info, seed_policy

_CONTEXTS = {
    "mnist": _c2_mnist_context,
    "fashion": _c2_fashion_context,
    "cifar10": _c2_cifar10_context,
}

# Geometric interior angles theta/pi.  Dense near the linear angle (1.0) where
# the theory predicts a collapse dip; endpoints (0.25, 1.25) excluded.
_DEFAULT_ANGLES_OVER_PI: tuple[float, ...] = (
    0.27,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,  # abs
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,  # relu
    0.80,
    0.85,
    0.90,
    0.93,
    0.95,
    0.97,
    0.98,
    0.99,
    1.00,  # linear (collapse)
    1.01,
    1.02,
    1.03,
    1.05,
    1.07,
    1.10,
    1.15,
    1.20,
    1.23,
)


def _angle_mix(theta: float) -> tuple[float, float]:
    """Return normalized (c1, c2) for interior angle ``theta`` (radians)."""
    b = math.tan(math.pi / 4 + theta)
    c1 = (1.0 + b) / 2.0
    c2 = (1.0 - b) / 2.0
    norm = math.hypot(c1, c2)
    return c1 / norm, c2 / norm


def _accumulate_blocks(
    *,
    enc: torch.Tensor,
    gain: torch.Tensor,
    bias: torch.Tensor,
    x_train: torch.Tensor,
    Y: torch.Tensor,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    n = enc.shape[0]
    d_out = Y.shape[1]
    ZtZ = torch.zeros(n, n, dtype=torch.float32)
    ZtA = torch.zeros(n, n, dtype=torch.float32)
    AtA = torch.zeros(n, n, dtype=torch.float32)
    ZtY = torch.zeros(n, d_out, dtype=torch.float32)
    AtY = torch.zeros(n, d_out, dtype=torch.float32)
    for s in range(0, x_train.shape[0], chunk_size):
        e = min(s + chunk_size, x_train.shape[0])
        z = x_train[s:e] @ enc.T
        z.mul_(gain)
        z.add_(bias)
        a = z.abs()
        ZtZ.addmm_(z.T, z)
        ZtA.addmm_(z.T, a)
        AtA.addmm_(a.T, a)
        yb = Y[s:e]
        ZtY.addmm_(z.T, yb)
        AtY.addmm_(a.T, yb)
    return {"ZtZ": ZtZ, "ZtA": ZtA, "AtA": AtA, "ZtY": ZtY, "AtY": AtY}


def _test_preactivation(
    *,
    enc: torch.Tensor,
    gain: torch.Tensor,
    bias: torch.Tensor,
    x_test: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for s in range(0, x_test.shape[0], chunk_size):
        e = min(s + chunk_size, x_test.shape[0])
        z = x_test[s:e] @ enc.T
        z.mul_(gain)
        z.add_(bias)
        chunks.append(z)
    return torch.cat(chunks, dim=0)


def _accuracy_for_angle(
    *,
    blocks: dict[str, torch.Tensor],
    Z_test: torch.Tensor,
    y_test: torch.Tensor,
    c1: float,
    c2: float,
    alpha: float,
    chunk_size: int,
) -> float:
    ata = (
        c1 * c1 * blocks["ZtZ"]
        + c1 * c2 * (blocks["ZtA"] + blocks["ZtA"].T)
        + c2 * c2 * blocks["AtA"]
    )
    aty = c1 * blocks["ZtY"] + c2 * blocks["AtY"]
    D = solve_from_normal_equations(ata, aty, alpha=(alpha, "trace"))
    n_correct = 0
    for s in range(0, Z_test.shape[0], chunk_size):
        e = min(s + chunk_size, Z_test.shape[0])
        z = Z_test[s:e]
        feat = c1 * z + c2 * z.abs()
        preds = (feat @ D).argmax(dim=1)
        n_correct += int((preds == y_test[s:e]).sum())
    return n_correct / Z_test.shape[0]


def run_dataset(
    *,
    dataset_key: str,
    seeds: list[int],
    angles_over_pi: tuple[float, ...],
    alpha: float,
    n_neurons: int,
    out_path: Path,
    results: dict,
) -> dict:
    ctx = _CONTEXTS[dataset_key]()
    chunk_size = 2500 if dataset_key == "cifar10" else 5000
    bundle = load_dataset_for_phase_c(ctx)
    x_train, y_train = bundle.x_train, bundle.y_train
    x_test, y_test = bundle.x_test, bundle.y_test
    d_in = int(x_train.shape[1])
    image_shape = tuple(ctx["image_shape"])
    Y = bundle.one_hot(y_train).float()

    for seed in seeds:
        t0 = time.perf_counter()
        set_sl_seed(seed)
        rngs = _make_subrngs_phase_c(seed)
        enc, gain, bias = build_encoders_and_biases(
            strategy="hypersphere",
            n_neurons=n_neurons,
            d_in=d_in,
            rng_encoder=rngs["encoder"],
            rng_gain=rngs["gain"],
            rng_center=rngs["center_index"],
            x_train=x_train,
            y_train=y_train,
            image_shape=image_shape,
            patch_size=5,
            gain_low=0.5,
            gain_high=2.0,
        )
        blocks = _accumulate_blocks(
            enc=enc, gain=gain, bias=bias, x_train=x_train, Y=Y, chunk_size=chunk_size
        )
        Z_test = _test_preactivation(
            enc=enc, gain=gain, bias=bias, x_test=x_test, chunk_size=chunk_size
        )
        build_s = time.perf_counter() - t0

        for tp in angles_over_pi:
            theta = tp * math.pi
            c1, c2 = _angle_mix(theta)
            acc = _accuracy_for_angle(
                blocks=blocks,
                Z_test=Z_test,
                y_test=y_test,
                c1=c1,
                c2=c2,
                alpha=alpha,
                chunk_size=chunk_size,
            )
            key = f"kink_angle_{dataset_key}_t{tp:.3f}_s{seed}"
            results[key] = {
                "dataset": dataset_key,
                "theta_over_pi": tp,
                "theta_rad": theta,
                "c1": c1,
                "c2": c2,
                "seed": seed,
                "n_neurons": n_neurons,
                "alpha": alpha,
                "test_accuracy": round(acc, 6),
                "code_identity": code_identity(),
                "seed_policy": seed_policy(),
                "hardware": hardware_info(),
                "build_time_s": round(build_s, 3),
            }
        _atomic_write(out_path, results)
        print(
            f"[kink_angle] {dataset_key} seed={seed} done "
            f"({len(angles_over_pi)} angles, build={build_s:.1f}s)",
            flush=True,
        )
        # Free the big buffers before the next seed.
        del blocks, Z_test
    return results


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--datasets", default="mnist,fashion,cifar10")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--n-neurons", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(t) for t in args.seeds.split(",") if t.strip()]
    dataset_keys = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for dataset_key in dataset_keys:
        out_path = (
            Path(args.out) if args.out else RESULTS_DIR / f"kink_angle_{dataset_key}_panel.json"
        )
        results = _load_json(out_path)
        print(
            f"[kink_angle] dataset={dataset_key} seeds={seeds} "
            f"code_identity={code_identity()} -> {out_path}",
            flush=True,
        )
        run_dataset(
            dataset_key=dataset_key,
            seeds=seeds,
            angles_over_pi=_DEFAULT_ANGLES_OVER_PI,
            alpha=args.alpha,
            n_neurons=args.n_neurons,
            out_path=out_path,
            results=results,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
