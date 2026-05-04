"""Experiment registry for ``benchmarks.cl_clean``.

Each entry maps an experiment name to:

* ``run_fn(config, seed, run_context) -> dict``: produce the metric
  fields for one ``(config, seed)`` cell.  The runner adds the
  reproducibility metadata and the sweep-axis fields (``knob``,
  ``value``) on top.
* ``default_run_context()``: the canonical run context for that
  experiment (e.g. base dataset, scenario, n_tasks).  Two cells can
  only be compared when their run contexts match.
* ``primary_metric`` / ``higher_is_better``: drives the analyzer's
  adoption decision and per-metric printing.

The first entry is a tiny synthetic smoke experiment used by the unit
tests.  Headline-claim experiments are added below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from leenef.layers import NEFLayer

from .cl_repro import set_cl_seed
from .datasets import make_task_stream

RunFn = Callable[[dict, int, dict], dict]
ContextFn = Callable[[], dict]


@dataclass(frozen=True)
class ExperimentSpec:
    """A registered CL clean experiment."""

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
# Smoke experiment: tiny synthetic 2-class CL stream.
# Purpose: keep the runner end-to-end tested in CI without pulling MNIST.
# ---------------------------------------------------------------------------


def _smoke_run(config: dict, seed: int, run_context: dict) -> dict:
    set_cl_seed(seed)
    n_tasks = int(run_context.get("n_tasks", 2))
    n_per_task = int(run_context.get("n_per_task", 64))
    dim = int(run_context.get("dim", 8))
    n_classes = int(run_context.get("n_classes", 4))

    g = torch.Generator().manual_seed(seed * 1_000_003 + 1)
    centers = torch.randn(n_classes, dim, generator=g) * 3.0

    def gen_task(task_classes: list[int], split_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        gg = torch.Generator().manual_seed(split_seed)
        n = n_per_task
        labels = torch.randint(0, len(task_classes), (n,), generator=gg)
        x = centers[torch.tensor([task_classes[i] for i in labels])] + 0.5 * torch.randn(
            n, dim, generator=gg
        )
        return x, torch.tensor([task_classes[i] for i in labels], dtype=torch.int64)

    classes_per_task = n_classes // n_tasks
    cls_perm = torch.randperm(n_classes, generator=g).tolist()
    task_classes = [
        cls_perm[t * classes_per_task : (t + 1) * classes_per_task] for t in range(n_tasks)
    ]

    layer = NEFLayer(
        d_in=dim,
        n_neurons=int(config["n_neurons"]),
        d_out=n_classes,
        encoder_strategy=str(config["encoder_strategy"]),
        activation=str(config["activation"]),
        gain=(float(config["gain_low"]), float(config["gain_high"])),
        rng=torch.Generator().manual_seed(seed),
    )
    alpha = float(config["alpha"])

    t0 = time.perf_counter()
    for t, cls in enumerate(task_classes):
        x_t, y_t = gen_task(cls, split_seed=seed * 7919 + t)
        oh = torch.zeros(x_t.shape[0], n_classes)
        oh[torch.arange(x_t.shape[0]), y_t] = 1.0
        layer.partial_fit(x_t, oh)
    layer.solve_accumulated(alpha=alpha)
    fit_s = time.perf_counter() - t0

    x_eval_parts, y_eval_parts = [], []
    for t, cls in enumerate(task_classes):
        x_t, y_t = gen_task(cls, split_seed=seed * 7919 + t + 1_000_000)
        x_eval_parts.append(x_t)
        y_eval_parts.append(y_t)
    x_eval = torch.cat(x_eval_parts)
    y_eval = torch.cat(y_eval_parts)

    with torch.no_grad():
        pred = layer(x_eval).argmax(dim=1)
        acc = float((pred == y_eval).float().mean())

    return {
        "test_accuracy": round(acc, 6),
        "fit_time_s": round(fit_s, 4),
        "n_tasks": n_tasks,
    }


def _smoke_context() -> dict:
    return {
        "experiment_kind": "synthetic_class_incremental",
        "n_tasks": 2,
        "n_per_task": 64,
        "dim": 8,
        "n_classes": 4,
    }


SMOKE_BASE_CONFIG: dict[str, object] = {
    "n_neurons": 200,
    "encoder_strategy": "hypersphere",
    "activation": "abs",
    "gain_low": 0.5,
    "gain_high": 2.0,
    "alpha": 1e-3,
}


register(
    ExperimentSpec(
        name="smoke",
        run_fn=_smoke_run,
        default_run_context=_smoke_context,
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=(
            "Tiny synthetic class-incremental smoke experiment used by "
            "the unit tests and as the worked example in docs/cl_clean_sweeps.md."
        ),
    )
)


# ---------------------------------------------------------------------------
# Headline NEF-CL experiment.
#
# Single-knob ``cell`` whose values enumerate (method × dataset × scenario).
# Methods at this stage:
#   - ``nef_accumulate``: sequential ``partial_fit``+``solve_accumulated``.
#   - ``nef_joint``: a single fit on the concatenation of all tasks
#     (oracle / upper bound).
#   - ``nef_reset``: refit only on the final task (forgetting lower bound).
#
# Datasets and scenarios live in ``run_context``; cells encode method only.
# This keeps the cell axis short while letting downstream sweeps split by
# dataset+scenario via separate experiment registrations.
#
# Implementation notes:
#   - Encoders + gains are derived from sub-RNGs depending only on the
#     seed, so the encoder is bit-identical across methods within one
#     seed (clean causal comparison).
#   - The center sub-RNG draws first-task centers (Paper II default —
#     "first_task" beats "all_tasks" within SE; see plan.md §4 H3).
#   - ``solve_accumulated`` uses trace-scaled Tikhonov regularization
#     by default; the α-regime panel sweeps ``alpha_mode`` between
#     ``trace`` and ``fixed``.
# ---------------------------------------------------------------------------


_NEF_CL_METHODS = ("nef_accumulate", "nef_joint", "nef_reset")
_MLP_METHODS = ("mlp_finetune", "mlp_joint", "mlp_ewc", "mlp_er")
NEF_CL_CELLS: tuple[str, ...] = _NEF_CL_METHODS
ALL_CL_CELLS: tuple[str, ...] = _NEF_CL_METHODS + _MLP_METHODS


def _make_subrngs(seed: int) -> dict[str, torch.Generator]:
    streams = ("encoder", "gain", "center_index", "task_stream")
    return {
        name: torch.Generator().manual_seed(seed * 1_000_003 + 17 * i + 1)
        for i, name in enumerate(streams)
    }


def _one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(y.shape[0], n_classes, dtype=torch.float32)
    oh[torch.arange(y.shape[0]), y] = 1.0
    return oh


def _build_layer(
    *,
    d_in: int,
    n_neurons: int,
    n_classes: int,
    encoder_strategy: str,
    gain_low: float,
    gain_high: float,
    centers: torch.Tensor | None,
    layer_seed: int,
    device: torch.device | None = None,
) -> NEFLayer:
    layer = NEFLayer(
        d_in=d_in,
        n_neurons=n_neurons,
        d_out=n_classes,
        encoder_strategy=encoder_strategy,
        activation="abs",
        gain=(gain_low, gain_high),
        centers=centers,
        rng=torch.Generator().manual_seed(layer_seed),
    )
    if device is not None:
        layer = layer.to(device)
    return layer


def _eval_per_task(
    layer: NEFLayer, stream, *, chunk: int = 5000, device: torch.device | None = None
) -> tuple[list[float], float]:
    per_task = []
    n_correct_total = 0
    n_total = 0
    with torch.no_grad():
        for task in stream.tasks:
            x, y = task.x_test, task.y_test
            n_correct = 0
            for s in range(0, x.shape[0], chunk):
                e = min(s + chunk, x.shape[0])
                xb = x[s:e]
                if device is not None:
                    xb = xb.to(device)
                pred = layer(xb).argmax(dim=1).cpu()
                n_correct += int((pred == y[s:e]).sum())
            per_task.append(round(n_correct / x.shape[0], 6))
            n_correct_total += n_correct
            n_total += x.shape[0]
    return per_task, round(n_correct_total / n_total, 6)


def _nef_cl_run(config: dict, seed: int, run_context: dict) -> dict:
    set_cl_seed(seed)
    method = str(config["cell"])
    if method in _MLP_METHODS:
        from .baselines import er_run, ewc_run, finetune_run, joint_run

        dispatch = {
            "mlp_finetune": finetune_run,
            "mlp_joint": joint_run,
            "mlp_ewc": ewc_run,
            "mlp_er": er_run,
        }
        return dispatch[method](config, seed, run_context)
    if method not in _NEF_CL_METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {ALL_CL_CELLS}")

    base_dataset = str(run_context["base_dataset"])
    scenario = str(run_context["scenario"])
    n_tasks = int(run_context["n_tasks"])

    rngs = _make_subrngs(seed)
    stream = make_task_stream(
        base_dataset,
        scenario=scenario,
        n_tasks=n_tasks,
        stream_seed=seed,
    )
    n_classes = stream.n_classes

    first_task = stream.tasks[0]
    d_in = int(first_task.x_train.shape[1])

    n_neurons = int(config["n_neurons"])
    alpha = float(config["alpha"])
    centers_mode = str(config.get("centers_mode", "first_task"))
    device_str = str(config.get("device", "cpu"))
    device = torch.device(device_str)

    if centers_mode == "first_task":
        idx = torch.randint(
            first_task.x_train.shape[0], (n_neurons,), generator=rngs["center_index"]
        )
        centers = first_task.x_train[idx]
    elif centers_mode == "all_tasks":
        all_x = torch.cat([t.x_train for t in stream.tasks], dim=0)
        idx = torch.randint(all_x.shape[0], (n_neurons,), generator=rngs["center_index"])
        centers = all_x[idx]
    elif centers_mode == "none":
        centers = None
    else:
        raise ValueError(f"unknown centers_mode {centers_mode!r}")
    if centers is not None:
        centers = centers.to(device)

    layer = _build_layer(
        d_in=d_in,
        n_neurons=n_neurons,
        n_classes=n_classes,
        encoder_strategy=str(config["encoder_strategy"]),
        gain_low=float(config["gain_low"]),
        gain_high=float(config["gain_high"]),
        centers=centers,
        layer_seed=seed * 1_000_003 + 1,
        device=device,
    )

    t0 = time.perf_counter()
    if method == "nef_accumulate":
        for task in stream.tasks:
            x = task.x_train.to(device)
            y_oh = _one_hot(task.y_train, n_classes).to(device)
            layer.partial_fit(x, y_oh)
        layer.solve_accumulated(alpha=alpha)
    elif method == "nef_joint":
        # Joint upper bound.  Concatenates the union once, then calls
        # ``partial_fit`` on shuffled chunks so the code path differs
        # from ``nef_accumulate`` (per-task in stream order) and the H1
        # joint-equivalence comparison is a real test rather than a
        # tautology.  Chunking caps the activity-matrix peak memory at
        # O(joint_chunk × n_neurons) instead of O(N_total × n_neurons).
        all_x = torch.cat([t.x_train for t in stream.tasks], dim=0)
        all_y = torch.cat([t.y_train for t in stream.tasks], dim=0)
        joint_chunk = int(config.get("joint_chunk_size", 4096))
        joint_g = torch.Generator().manual_seed(seed * 1_000_003 + 7919)
        order = torch.randperm(all_x.shape[0], generator=joint_g)
        for s in range(0, all_x.shape[0], joint_chunk):
            sel = order[s : s + joint_chunk]
            xb = all_x[sel].to(device)
            yb = _one_hot(all_y[sel], n_classes).to(device)
            layer.partial_fit(xb, yb)
        layer.solve_accumulated(alpha=alpha)
    elif method == "nef_reset":
        last = stream.tasks[-1]
        x = last.x_train.to(device)
        y_oh = _one_hot(last.y_train, n_classes).to(device)
        layer.partial_fit(x, y_oh)
        layer.solve_accumulated(alpha=alpha)
    fit_s = time.perf_counter() - t0

    per_task_acc, avg_acc = _eval_per_task(layer, stream, device=device)

    return {
        "test_accuracy": avg_acc,
        "per_task_accuracy": per_task_acc,
        "fit_time_s": round(fit_s, 4),
        "stream_hash": stream.stream_hash,
        "stream_metadata": stream.metadata(),
        "dataset_metadata": {
            "base_dataset": stream.base_name,
            "scenario": stream.scenario,
            "n_tasks": int(stream.n_tasks),
            "n_classes": int(stream.n_classes),
            "stream_seed": int(stream.stream_seed),
            "stream_hash": stream.stream_hash,
            "raw_hashes": dict(stream.raw_hashes),
            "processed_hashes": dict(stream.processed_hashes),
        },
    }


def _nef_cl_context_factory(*, base_dataset: str, scenario: str, n_tasks: int) -> ContextFn:
    def _ctx() -> dict:
        return {
            "experiment_kind": "nef_continual",
            "base_dataset": base_dataset,
            "scenario": scenario,
            "n_tasks": n_tasks,
        }

    return _ctx


# Headline panels: (registered_name, base_dataset, scenario, n_tasks).
_NEF_CL_PANELS: tuple[tuple[str, str, str, int], ...] = (
    ("nef_continual_split_mnist", "MNIST", "split", 5),
    ("nef_continual_permuted_mnist", "MNIST", "permuted", 5),
    ("nef_continual_split_cifar10", "CIFAR10", "split", 5),
    ("nef_continual_split_cifar100", "CIFAR100", "split", 10),
)

for _name, _base, _scen, _nt in _NEF_CL_PANELS:
    register(
        ExperimentSpec(
            name=_name,
            run_fn=_nef_cl_run,
            default_run_context=_nef_cl_context_factory(
                base_dataset=_base, scenario=_scen, n_tasks=_nt
            ),
            primary_metric="test_accuracy",
            higher_is_better=True,
            description=(
                f"Paper II headline NEF-CL panel — {_base} ({_scen}, {_nt} tasks). "
                f"Cells: {NEF_CL_CELLS}."
            ),
        )
    )
