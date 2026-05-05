"""ConvNEF continual-learning experiment for Paper II Phase C.

The experiment keeps the ``cl_clean`` identity discipline intact:

* the actual experiment logic lives under ``benchmarks/cl_clean/`` so
  GPU rows carry a meaningful ``code_identity``;
* the Colab package only rewrites configs and invokes the standard
  phase runner;
* the result schema mirrors the existing NEF-CL panels
  (``dataset_metadata``, ``stream_metadata``, per-task accuracies).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from leenef._rng import fork_generator
from leenef.conv import ConvNEFPipeline
from leenef.layers import NEFLayer

from .cl_repro import set_cl_seed
from .datasets import TaskStream, make_task_stream

DEFAULT_CONVNEF_STAGES: tuple[dict[str, int], ...] = (
    {"n_filters": 32, "patch_size": 3, "pool_size": 1},
    {"n_filters": 32, "patch_size": 5, "pool_size": 1},
    {"n_filters": 32, "patch_size": 7, "pool_size": 1},
)

PHASE_C_CONVNEF_CELLS: tuple[str, ...] = (
    "all_data_n10000",
    "first_task_n10000",
    "joint_n10000",
    "all_data_n20000",
    "first_task_n20000",
    "joint_n20000",
)


@dataclass
class ConvNEFCLResult:
    method: str
    feature_mode: str
    n_neurons: int
    n_tasks: int
    accuracy_matrix: list[list[float]]
    fit_times: list[float]
    feature_time: float
    config: dict = field(default_factory=dict)

    @property
    def final_avg_accuracy(self) -> float:
        return sum(self.accuracy_matrix[-1]) / len(self.accuracy_matrix[-1])

    @property
    def forgetting(self) -> float:
        n_rows = len(self.accuracy_matrix)
        if n_rows <= 1:
            return 0.0
        n = len(self.accuracy_matrix[-1])
        total = 0.0
        for j in range(n):
            peak = max(self.accuracy_matrix[i][j] for i in range(n_rows))
            total += peak - self.accuracy_matrix[-1][j]
        return total / n

    @property
    def total_time(self) -> float:
        return self.feature_time + sum(self.fit_times)


def parse_convnef_cell(raw: object) -> tuple[str, int]:
    """Parse labels like ``all_data_n10000`` or ``joint_n20000``."""
    text = str(raw)
    try:
        mode, n_text = text.rsplit("_n", 1)
        n_neurons = int(n_text)
    except ValueError as exc:
        raise ValueError(
            "convnef cell must look like 'all_data_n10000', "
            f"'first_task_n20000', or 'joint_n10000'; got {text!r}"
        ) from exc
    valid = {"all_data", "first_task", "joint"}
    if mode not in valid:
        raise ValueError(f"unknown ConvNEF cell mode {mode!r}; expected one of {sorted(valid)}")
    if n_neurons <= 0:
        raise ValueError(f"ConvNEF cell neuron count must be positive, got {n_neurons}")
    return mode, n_neurons


def _one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(y, n_classes).to(torch.float32)


def _reshape_flat_cifar10(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != 3072:
        raise ValueError(f"expected flat CIFAR-10 tensor of shape (N, 3072), got {tuple(x.shape)}")
    return x.reshape(x.shape[0], 3, 32, 32).contiguous()


def _stream_to_image_tasks(stream: TaskStream, *, device: torch.device) -> list[dict]:
    tasks: list[dict] = []
    for task in stream.tasks:
        tasks.append(
            {
                "name": f"task_{task.task_index}",
                "classes": list(task.classes),
                "x_train": _reshape_flat_cifar10(task.x_train).to(device),
                "y_train": task.y_train.to(device),
                "x_test": _reshape_flat_cifar10(task.x_test).to(device),
                "y_test": task.y_test.to(device),
            }
        )
    return tasks


def _build_pipeline(config: dict, *, n_neurons: int) -> ConvNEFPipeline:
    head_kwargs = {
        "encoder_strategy": str(config.get("encoder_strategy", "hypersphere")),
        "activation": str(config.get("activation", "abs")),
        "gain": (float(config.get("gain_low", 0.5)), float(config.get("gain_high", 2.0))),
    }
    return ConvNEFPipeline(
        stages=list(config.get("conv_stages", DEFAULT_CONVNEF_STAGES)),
        n_neurons=n_neurons,
        pool_levels=list(config.get("pool_levels", [1, 2, 4])),
        pool_order=int(config.get("pool_order", 1)),
        standardize=bool(config.get("standardize", True)),
        gcn=bool(config.get("gcn", False)),
        lcn_kernel=config.get("lcn_kernel"),
        parallel=bool(config.get("parallel", True)),
        **head_kwargs,
    )


def _fit_pipeline_features(
    pipeline: ConvNEFPipeline,
    images: torch.Tensor,
    n_classes: int,
    *,
    fit_subsample: int,
    batch_size: int,
    seed: int,
) -> None:
    """Fit ConvNEF stages and create the NEF head without solving decoders."""
    n_samples = images.shape[0]
    g = torch.Generator(device=images.device).manual_seed(seed)

    sub_n = min(fit_subsample, n_samples)
    sub_idx = torch.randperm(n_samples, generator=g, device=images.device)[:sub_n]
    sub_images = pipeline._preprocess(images[sub_idx])

    if pipeline.parallel:
        for stage in pipeline.stages:
            stage.fit(sub_images, rng=fork_generator(g, device=sub_images.device))
    else:
        x_sub = sub_images
        for stage in pipeline.stages:
            stage.fit(x_sub, rng=fork_generator(g, device=x_sub.device))
            chunks = []
            for start in range(0, x_sub.shape[0], batch_size):
                chunks.append(stage(x_sub[start : start + batch_size]))
            x_sub = torch.cat(chunks, dim=0)
            del chunks

    center_chunks = []
    for start in range(0, sub_n, batch_size):
        chunk = pipeline._apply_stages(sub_images[start : start + batch_size])
        center_chunks.append(pipeline._pool_features(chunk))
        del chunk
    x_sub = torch.cat(center_chunks, dim=0)
    del center_chunks

    centers = pipeline._standardize_features(x_sub, fit=True)
    del x_sub
    feat_dim = centers.shape[1]

    head_kwargs = dict(pipeline.nef_kwargs)
    strategy = head_kwargs.get("encoder_strategy", "hypersphere")
    data_strategies = {"whitened", "class_contrast", "local_pca"}
    if strategy in data_strategies:
        encoder_kwargs = dict(head_kwargs.get("encoder_kwargs", {}) or {})
        encoder_kwargs.setdefault("train_data", centers)
        head_kwargs["encoder_kwargs"] = encoder_kwargs

    pipeline.head = NEFLayer(
        feat_dim,
        pipeline.n_neurons,
        n_classes,
        centers=centers,
        rng=fork_generator(g, device=images.device),
        **head_kwargs,
    ).to(images.device)


@torch.no_grad()
def _extract_features(
    pipeline: ConvNEFPipeline,
    images: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        x = pipeline._preprocess(images[start : start + batch_size])
        x = pipeline._apply_stages(x)
        x = pipeline._pool_features(x)
        x = pipeline._standardize_features(x)
        chunks.append(x)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _evaluate_pipeline_on_tasks(
    pipeline: ConvNEFPipeline,
    tasks: list[dict],
    *,
    batch_size: int,
) -> list[float]:
    pipeline.eval()
    out = []
    for task in tasks:
        pred = pipeline.predict(task["x_test"], batch_size=batch_size)
        out.append(float((pred.argmax(1) == task["y_test"]).float().mean()))
    return out


def _run_convnef_cl(
    tasks: list[dict],
    feature_images: torch.Tensor,
    feature_mode: str,
    *,
    config: dict,
    n_classes: int,
    n_neurons: int,
    seed: int,
) -> ConvNEFCLResult:
    batch_size = int(config.get("batch_size", 1000))
    fit_subsample = int(config.get("fit_subsample", 10_000))
    alpha = float(config.get("alpha", 1e-2))
    augment_flip = bool(config.get("augment_flip", True))
    n_augment = int(config.get("n_augment", 1))

    pipeline = _build_pipeline(config, n_neurons=n_neurons)

    t0 = time.perf_counter()
    _fit_pipeline_features(
        pipeline,
        feature_images,
        n_classes,
        fit_subsample=fit_subsample,
        batch_size=batch_size,
        seed=seed,
    )
    feature_time = time.perf_counter() - t0

    pipeline.head.reset_accumulators()
    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []
    augment_fn = (lambda x: x.flip(-1)) if augment_flip else None

    for task in tasks:
        targets = _one_hot(task["y_train"], n_classes)
        t0 = time.perf_counter()
        for start in range(0, task["x_train"].shape[0], batch_size):
            x_batch = task["x_train"][start : start + batch_size]
            t_batch = targets[start : start + batch_size]
            features = _extract_features(pipeline, x_batch, batch_size=batch_size)
            pipeline.head.partial_fit(features, t_batch)
            del features

            if augment_fn is not None:
                for _ in range(n_augment):
                    x_aug = augment_fn(x_batch)
                    f_aug = _extract_features(pipeline, x_aug, batch_size=batch_size)
                    pipeline.head.partial_fit(f_aug, t_batch)
                    del f_aug

        pipeline.head.solve_accumulated(alpha=alpha)
        fit_times.append(time.perf_counter() - t0)
        accuracy_matrix.append(_evaluate_pipeline_on_tasks(pipeline, tasks, batch_size=batch_size))

    return ConvNEFCLResult(
        method=f"ConvNEF-CL ({feature_mode})",
        feature_mode=feature_mode,
        n_neurons=n_neurons,
        n_tasks=len(tasks),
        accuracy_matrix=accuracy_matrix,
        fit_times=fit_times,
        feature_time=feature_time,
        config={
            "alpha": alpha,
            "conv_stages": config.get("conv_stages", DEFAULT_CONVNEF_STAGES),
            "pool_levels": config.get("pool_levels", [1, 2, 4]),
            "pool_order": int(config.get("pool_order", 1)),
            "standardize": bool(config.get("standardize", True)),
            "gcn": bool(config.get("gcn", False)),
            "parallel": bool(config.get("parallel", True)),
            "augment_flip": augment_flip,
            "n_augment": n_augment,
        },
    )


def _run_convnef_joint(
    all_train_images: torch.Tensor,
    all_train_labels: torch.Tensor,
    tasks: list[dict],
    *,
    config: dict,
    n_classes: int,
    n_neurons: int,
    seed: int,
) -> ConvNEFCLResult:
    batch_size = int(config.get("batch_size", 1000))
    fit_subsample = int(config.get("fit_subsample", 10_000))
    alpha = float(config.get("alpha", 1e-2))
    augment_flip = bool(config.get("augment_flip", True))
    n_augment = int(config.get("n_augment", 1))

    pipeline = _build_pipeline(config, n_neurons=n_neurons)
    targets = _one_hot(all_train_labels, n_classes)
    augment_fn = (lambda x: x.flip(-1)) if augment_flip else None

    t0 = time.perf_counter()
    pipeline.fit(
        all_train_images,
        targets,
        alpha=alpha,
        fit_subsample=fit_subsample,
        batch_size=batch_size,
        seed=seed,
        augment_fn=augment_fn,
        n_augment=n_augment,
    )
    total_time = time.perf_counter() - t0
    accuracy = _evaluate_pipeline_on_tasks(pipeline, tasks, batch_size=batch_size)

    return ConvNEFCLResult(
        method="ConvNEF-joint",
        feature_mode="joint",
        n_neurons=n_neurons,
        n_tasks=len(tasks),
        accuracy_matrix=[accuracy],
        fit_times=[total_time],
        feature_time=0.0,
        config={
            "alpha": alpha,
            "conv_stages": config.get("conv_stages", DEFAULT_CONVNEF_STAGES),
            "pool_levels": config.get("pool_levels", [1, 2, 4]),
            "pool_order": int(config.get("pool_order", 1)),
            "standardize": bool(config.get("standardize", True)),
            "gcn": bool(config.get("gcn", False)),
            "parallel": bool(config.get("parallel", True)),
            "augment_flip": augment_flip,
            "n_augment": n_augment,
        },
    )


def convnef_cl_run(config: dict, seed: int, run_context: dict) -> dict:
    """Run the Split-CIFAR-10 ConvNEF continual-learning panel."""
    set_cl_seed(seed)
    config = dict(config)
    run_context = dict(run_context)
    feature_mode, n_neurons = parse_convnef_cell(config["cell"])

    if (
        str(run_context.get("base_dataset")) != "CIFAR10"
        or str(run_context.get("scenario")) != "split"
    ):
        raise ValueError(
            "ConvNEF Phase C expects base_dataset='CIFAR10' and scenario='split', "
            f"got {run_context!r}"
        )

    device = torch.device(str(config.get("device", "cpu")))
    stream = make_task_stream(
        "CIFAR10",
        scenario="split",
        n_tasks=int(run_context.get("n_tasks", 5)),
        stream_seed=seed,
    )
    tasks = _stream_to_image_tasks(stream, device=device)
    all_train_images = torch.cat([task["x_train"] for task in tasks], dim=0)
    all_train_labels = torch.cat([task["y_train"] for task in tasks], dim=0)

    if feature_mode == "joint":
        result = _run_convnef_joint(
            all_train_images,
            all_train_labels,
            tasks,
            config=config,
            n_classes=stream.n_classes,
            n_neurons=n_neurons,
            seed=seed,
        )
    else:
        feature_images = all_train_images if feature_mode == "all_data" else tasks[0]["x_train"]
        result = _run_convnef_cl(
            tasks,
            feature_images,
            feature_mode,
            config=config,
            n_classes=stream.n_classes,
            n_neurons=n_neurons,
            seed=seed,
        )

    per_task_accuracy = [round(v, 6) for v in result.accuracy_matrix[-1]]
    accuracy_matrix = [[round(v, 6) for v in row] for row in result.accuracy_matrix]
    return {
        "test_accuracy": round(result.final_avg_accuracy, 6),
        "per_task_accuracy": per_task_accuracy,
        "accuracy_matrix": accuracy_matrix,
        "fit_time_s": round(result.total_time, 4),
        "feature_time_s": round(result.feature_time, 4),
        "task_fit_times_s": [round(v, 4) for v in result.fit_times],
        "forgetting": round(result.forgetting, 6),
        "effective_n_neurons": n_neurons,
        "feature_mode": feature_mode,
        "alpha": float(config.get("alpha", 1e-2)),
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
