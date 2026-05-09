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
from leenef.solvers import solve_from_normal_equations

from ._c1_convnef_cl_exp import PHASE_C_CONVNEF_CELLS, convnef_cl_run
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


def _make_gain_tensor(
    n_neurons: int, gain_low: float, gain_high: float, *, rng: torch.Generator
) -> torch.Tensor:
    return gain_low + (gain_high - gain_low) * torch.rand(n_neurons, generator=rng)


def _one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    oh = torch.zeros(y.shape[0], n_classes, dtype=torch.float32)
    oh[torch.arange(y.shape[0]), y] = 1.0
    return oh


def _parse_capacity_cell(raw: object) -> tuple[int, int]:
    """Parse capacity labels of the form ``n5000_t20``."""
    text = str(raw)
    try:
        n_part, t_part = text.split("_", 1)
        if not n_part.startswith("n") or not t_part.startswith("t"):
            raise ValueError
        return int(n_part[1:]), int(t_part[1:])
    except ValueError as exc:
        raise ValueError(f"capacity_cell must look like 'n5000_t20', got {text!r}") from exc


def _parse_solver_alpha_cell(raw: object) -> tuple[str, float]:
    """Parse solver labels of the form ``woodbury_online_a0.1``."""
    text = str(raw)
    try:
        solver_mode, alpha_text = text.rsplit("_a", 1)
        alpha = float(alpha_text)
    except ValueError as exc:
        raise ValueError(
            "solver_alpha_cell must look like 'accumulate_a0.001' or "
            f"'woodbury_online_a1', got {text!r}"
        ) from exc
    valid = {"accumulate", "woodbury_batch", "woodbury_online"}
    if solver_mode not in valid:
        raise ValueError(f"unknown solver mode {solver_mode!r}; expected one of {sorted(valid)}")
    return solver_mode, alpha


def _normalize_centers_mode(raw: object) -> str:
    return "none" if raw is None else str(raw)


def _build_layer(
    *,
    d_in: int,
    n_neurons: int,
    n_classes: int,
    encoder_strategy: str,
    gain_low: float,
    gain_high: float,
    centers: torch.Tensor | None,
    rngs: dict[str, torch.Generator],
    device: torch.device | None = None,
) -> NEFLayer:
    gain = _make_gain_tensor(
        n_neurons,
        gain_low,
        gain_high,
        rng=rngs["gain"],
    )
    layer = NEFLayer(
        d_in=d_in,
        n_neurons=n_neurons,
        d_out=n_classes,
        encoder_strategy=encoder_strategy,
        activation="abs",
        gain=gain,
        centers=None,
        rng=rngs["encoder"],
    )
    if centers is not None:
        if centers.shape != (n_neurons, d_in):
            raise ValueError(
                f"selected centers must have shape {(n_neurons, d_in)}, got {tuple(centers.shape)}"
            )
        with torch.no_grad():
            layer.bias.data.copy_(-layer.gain * (centers.float() * layer.encoders.data).sum(dim=1))
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
            y = task.y_test
            n_correct = 0
            for s in range(0, y.shape[0], chunk):
                e = min(s + chunk, y.shape[0])
                xb = task.test_inputs(start=s, end=e)
                if device is not None:
                    xb = xb.to(device)
                pred = layer(xb).argmax(dim=1).cpu()
                n_correct += int((pred == y[s:e]).sum())
            per_task.append(round(n_correct / y.shape[0], 6))
            n_correct_total += n_correct
            n_total += y.shape[0]
    return per_task, round(n_correct_total / n_total, 6)


def _sample_all_task_centers(stream, *, n_neurons: int, rng: torch.Generator) -> torch.Tensor:
    first_task = stream.tasks[0]
    d_in = int(first_task.x_train.shape[1])
    counts = torch.tensor([task.x_train.shape[0] for task in stream.tasks], dtype=torch.int64)
    total = int(counts.sum().item())
    draw = torch.randint(total, (n_neurons,), generator=rng)
    offsets = torch.cumsum(counts, dim=0)
    centers = torch.empty(n_neurons, d_in, dtype=first_task.x_train.dtype)
    prev = 0
    for task_idx, task in enumerate(stream.tasks):
        end = int(offsets[task_idx].item())
        mask = (draw >= prev) & (draw < end)
        if mask.any():
            local = draw[mask] - prev
            centers[mask] = task.train_inputs(indices=local)
        prev = end
    return centers


def _fit_joint_shuffled_chunks(
    layer: NEFLayer,
    stream,
    *,
    n_classes: int,
    joint_chunk: int,
    rng: torch.Generator,
    device: torch.device,
) -> None:
    chunks: list[tuple[int, int, int]] = []
    for task_idx, task in enumerate(stream.tasks):
        for start in range(0, task.x_train.shape[0], joint_chunk):
            end = min(start + joint_chunk, task.x_train.shape[0])
            chunks.append((task_idx, start, end))

    order = torch.randperm(len(chunks), generator=rng)
    for idx in order.tolist():
        task_idx, start, end = chunks[idx]
        task = stream.tasks[task_idx]
        xb = task.train_inputs(start=start, end=end).to(device)
        yb = _one_hot(task.y_train[start:end], n_classes).to(device)
        layer.partial_fit(xb, yb)


def _fit_accumulate_chunked(
    layer: NEFLayer,
    stream,
    *,
    n_classes: int,
    chunk_size: int,
    device: torch.device,
    mode: str,
    alpha: float,
) -> None:
    for task in stream.tasks:
        y_oh_full = _one_hot(task.y_train, n_classes)
        for start in range(0, task.x_train.shape[0], chunk_size):
            end = min(start + chunk_size, task.x_train.shape[0])
            x = task.train_inputs(start=start, end=end).to(device)
            y_oh = y_oh_full[start:end].to(device)
            if mode == "accumulate":
                layer.partial_fit(x, y_oh)
            elif mode == "woodbury_batch":
                layer.continuous_fit(x, y_oh, alpha=alpha)
            else:
                raise ValueError(f"unsupported chunked fit mode {mode!r}")


@torch.no_grad()
def _fit_growing_centers(
    layer: NEFLayer,
    stream,
    *,
    n_classes: int,
    alpha: float,
    rng: torch.Generator,
    device: torch.device,
) -> None:
    """Fit the negative-control "growing neuron pool" center strategy.

    Neurons are partitioned by task.  When task ``t`` arrives, its group
    receives centers sampled from that task and becomes active; future
    groups are zeroed out for the current task.  This preserves the old
    no-replay design from the checkpoint trail: old-data × new-neuron
    cross-terms are missing, so the panel tests whether the tempting
    center-adaptation shortcut actually helps (it previously collapsed).
    """

    n_tasks = len(stream.tasks)
    boundaries = torch.linspace(0, layer.n_neurons, n_tasks + 1, dtype=torch.int64).tolist()
    ata = torch.zeros(layer.n_neurons, layer.n_neurons, dtype=torch.float64, device=device)
    aty = torch.zeros(layer.n_neurons, n_classes, dtype=torch.float64, device=device)

    for t, task in enumerate(stream.tasks):
        start, end = int(boundaries[t]), int(boundaries[t + 1])
        if end <= start:
            raise ValueError(
                f"growing centers need at least one neuron per task; "
                f"n_neurons={layer.n_neurons}, n_tasks={n_tasks}"
            )
        idx = torch.randint(task.x_train.shape[0], (end - start,), generator=rng)
        centers = task.train_inputs(indices=idx).to(device)
        with torch.no_grad():
            enc = layer.encoders[start:end]
            gain = layer.gain[start:end]
            layer.bias[start:end].copy_(-gain * (centers.float() * enc).sum(dim=1))

        x = task.train_inputs().to(device)
        y_oh = _one_hot(task.y_train, n_classes).to(device)
        acts = layer.encode(x)
        if end < layer.n_neurons:
            acts[:, end:] = 0
        acts_stats = acts.float() if acts.dtype in (torch.float16, torch.bfloat16) else acts
        y_stats = y_oh.float() if y_oh.dtype in (torch.float16, torch.bfloat16) else y_oh
        ata.add_((acts_stats.T @ acts_stats).double())
        aty.add_((acts_stats.T @ y_stats).double())

    layer._ata = ata
    layer._aty = aty
    layer.decoders.data.copy_(
        solve_from_normal_equations(layer._ata, layer._aty, alpha=alpha).to(layer.decoders.dtype)
    )


def _nef_cl_run(config: dict, seed: int, run_context: dict) -> dict:
    set_cl_seed(seed)
    config = dict(config)
    run_context = dict(run_context)
    if "capacity_cell" in config:
        n_neurons_override, n_tasks_override = _parse_capacity_cell(config["capacity_cell"])
        config["n_neurons"] = n_neurons_override
        run_context["n_tasks"] = n_tasks_override
        config.setdefault("cell", "nef_accumulate")
    if "solver_alpha_cell" in config:
        solver_mode_override, alpha_override = _parse_solver_alpha_cell(
            config["solver_alpha_cell"]
        )
        config["solver_mode"] = solver_mode_override
        config["alpha"] = alpha_override
        config.setdefault("cell", "nef_accumulate")

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
    centers_mode = _normalize_centers_mode(config.get("centers_mode", "first_task"))
    solver_mode = str(config.get("solver_mode", "accumulate"))
    fit_chunk_size = int(config.get("fit_chunk_size", 4096))
    device_str = str(config.get("device", "cpu"))
    device = torch.device(device_str)

    if centers_mode == "first_task":
        idx = torch.randint(
            first_task.x_train.shape[0], (n_neurons,), generator=rngs["center_index"]
        )
        centers = first_task.train_inputs(indices=idx)
    elif centers_mode == "all_tasks":
        centers = _sample_all_task_centers(stream, n_neurons=n_neurons, rng=rngs["center_index"])
    elif centers_mode == "none":
        centers = None
    elif centers_mode == "growing":
        centers = None
    else:
        raise ValueError(f"unknown centers_mode {centers_mode!r}")
    layer = _build_layer(
        d_in=d_in,
        n_neurons=n_neurons,
        n_classes=n_classes,
        encoder_strategy=str(config["encoder_strategy"]),
        gain_low=float(config["gain_low"]),
        gain_high=float(config["gain_high"]),
        centers=centers,
        rngs=rngs,
        device=device,
    )

    refresh_decoder_max_abs_diff: float | None = None
    refresh_test_accuracy: float | None = None

    t0 = time.perf_counter()
    if method == "nef_accumulate" and centers_mode == "growing":
        _fit_growing_centers(
            layer,
            stream,
            n_classes=n_classes,
            alpha=alpha,
            rng=rngs["center_index"],
            device=device,
        )
    elif method == "nef_accumulate" and solver_mode == "accumulate":
        _fit_accumulate_chunked(
            layer,
            stream,
            n_classes=n_classes,
            chunk_size=fit_chunk_size,
            device=device,
            mode="accumulate",
            alpha=alpha,
        )
        layer.solve_accumulated(alpha=alpha)
    elif method == "nef_accumulate" and solver_mode == "woodbury_batch":
        _fit_accumulate_chunked(
            layer,
            stream,
            n_classes=n_classes,
            chunk_size=fit_chunk_size,
            device=device,
            mode="woodbury_batch",
            alpha=alpha,
        )
    elif method == "nef_accumulate" and solver_mode == "woodbury_online":
        chunk = int(config.get("woodbury_chunk_size", 100))
        for task in stream.tasks:
            y_oh_full = _one_hot(task.y_train, n_classes)
            for s in range(0, task.x_train.shape[0], chunk):
                e = min(s + chunk, task.x_train.shape[0])
                x = task.train_inputs(start=s, end=e).to(device)
                y_oh = y_oh_full[s:e].to(device)
                layer.continuous_fit(x, y_oh, alpha=alpha)
        online_decoders = layer.decoders.data.detach().clone()
        layer.refresh_inverse(alpha=alpha)
        refresh_decoder_max_abs_diff = float(
            (online_decoders - layer.decoders.data).abs().max().cpu()
        )
        refresh_per_task_acc, refresh_test_accuracy = _eval_per_task(layer, stream, device=device)
        layer.decoders.data.copy_(online_decoders)
    elif method == "nef_accumulate":
        raise ValueError(
            f"unknown solver_mode {solver_mode!r}; expected accumulate, "
            "woodbury_batch, or woodbury_online"
        )
    elif method == "nef_joint":
        # Joint upper bound.  Feed the union through ``partial_fit`` in a
        # globally shuffled chunk order without materializing the full
        # concatenated dataset, which keeps large B3 permuted grids from
        # allocating one all-tasks tensor in RAM.
        joint_chunk = int(config.get("joint_chunk_size", 4096))
        joint_g = torch.Generator().manual_seed(seed * 1_000_003 + 7919)
        _fit_joint_shuffled_chunks(
            layer,
            stream,
            n_classes=n_classes,
            joint_chunk=joint_chunk,
            rng=joint_g,
            device=device,
        )
        layer.solve_accumulated(alpha=alpha)
    elif method == "nef_reset":
        last = stream.tasks[-1]
        x = last.train_inputs().to(device)
        y_oh = _one_hot(last.y_train, n_classes).to(device)
        layer.partial_fit(x, y_oh)
        layer.solve_accumulated(alpha=alpha)
    fit_s = time.perf_counter() - t0

    per_task_acc, avg_acc = _eval_per_task(layer, stream, device=device)

    metrics = {
        "test_accuracy": avg_acc,
        "per_task_accuracy": per_task_acc,
        "fit_time_s": round(fit_s, 4),
        "effective_n_neurons": n_neurons,
        "effective_n_tasks": n_tasks,
        "centers_mode": centers_mode,
        "solver_mode": solver_mode,
        "alpha": alpha,
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
    if refresh_decoder_max_abs_diff is not None:
        metrics["refresh_decoder_max_abs_diff"] = refresh_decoder_max_abs_diff
        metrics["refresh_test_accuracy"] = refresh_test_accuracy
        metrics["refresh_per_task_accuracy"] = refresh_per_task_acc
    return metrics


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


# Phase B3 capacity panels.  These are separate experiment registrations
# rather than run-context overrides so each row's top-level run_context
# records the actual task count used by the phase.
for _nt in (5, 10, 20, 50, 100):
    register(
        ExperimentSpec(
            name=f"nef_capacity_permuted_mnist_t{_nt}",
            run_fn=_nef_cl_run,
            default_run_context=_nef_cl_context_factory(
                base_dataset="MNIST", scenario="permuted", n_tasks=_nt
            ),
            primary_metric="test_accuracy",
            higher_is_better=True,
            description=f"Paper II B3 capacity panel — Permuted-MNIST ({_nt} tasks).",
        )
    )

for _nt in (5, 10, 20):
    register(
        ExperimentSpec(
            name=f"nef_capacity_permuted_cifar10_t{_nt}",
            run_fn=_nef_cl_run,
            default_run_context=_nef_cl_context_factory(
                base_dataset="CIFAR10", scenario="permuted", n_tasks=_nt
            ),
            primary_metric="test_accuracy",
            higher_is_better=True,
            description=f"Paper II B3 capacity spot panel — Permuted-CIFAR-10 ({_nt} tasks).",
        )
    )


register(
    ExperimentSpec(
        name="c1_convnef_split_cifar10",
        run_fn=convnef_cl_run,
        default_run_context=_nef_cl_context_factory(
            base_dataset="CIFAR10", scenario="split", n_tasks=5
        ),
        primary_metric="test_accuracy",
        higher_is_better=True,
        description=(
            "Paper II Phase C ConvNEF continual-learning panel on Split-CIFAR-10. "
            f"Cells: {PHASE_C_CONVNEF_CELLS}."
        ),
    )
)
