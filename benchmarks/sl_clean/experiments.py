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
experiments will be added as separate entries during phases B and C.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from leenef.layers import NEFLayer

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
