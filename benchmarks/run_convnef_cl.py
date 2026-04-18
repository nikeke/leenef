"""ConvNEF + continual learning benchmark.

Tests whether gradient-free convolutional features (PCA filters) can serve
as a fixed encoder for continual analytical decoding on CIFAR-10.

Feature learning modes
----------------------
- **all_data**: PCA filters learned from *all* training images (oracle
  features, upper bound on feature quality).
- **first_task**: PCA filters learned from first task only.  PCA is
  class-agnostic, so first-task features may transfer well.

Both modes freeze the ConvNEF pipeline after feature learning and run
continual analytical decoding via ``partial_fit`` + ``solve_accumulated``
on Split-CIFAR-10 (5 tasks, 2 classes each).

Baselines
---------
- **ConvNEF-joint**: non-CL ConvNEF fit on all data at once (ceiling).
- **Flat NEF CL**: raw-pixel NEF accumulation (floor from prior work).

Usage::

    python benchmarks/run_convnef_cl.py --seed 0
    python benchmarks/run_convnef_cl.py --n-neurons 20000 --device cuda --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from leenef.conv import ConvNEFPipeline  # noqa: E402
from leenef.layers import NEFLayer  # noqa: E402

# ── Data loading ──────────────────────────────────────────────────────


def load_cifar10_images(root: str = "./data", device: str = "cpu"):
    """Load CIFAR-10 as (N,3,32,32) image tensors on the target device."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)
    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds]).to(device)
    x_test = torch.stack([img for img, _ in test_ds]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_ds]).to(device)
    return x_train, y_train, x_test, y_test


# ── Task construction ─────────────────────────────────────────────────


def split_cifar10_tasks(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    classes_per_task: int = 2,
) -> list[dict]:
    """Create Split-CIFAR-10 class-incremental tasks."""
    n_classes = 10
    n_tasks = n_classes // classes_per_task
    tasks = []
    for t in range(n_tasks):
        lo = t * classes_per_task
        hi = (t + 1) * classes_per_task
        cls = list(range(lo, hi))
        tr_mask = sum(y_train == c for c in cls).bool()
        te_mask = sum(y_test == c for c in cls).bool()
        tasks.append(
            {
                "name": f"classes {cls}",
                "classes": cls,
                "x_train": x_train[tr_mask],
                "y_train": y_train[tr_mask],
                "x_test": x_test[te_mask],
                "y_test": y_test[te_mask],
            }
        )
    return tasks


# ── Result container ──────────────────────────────────────────────────


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
            final = self.accuracy_matrix[-1][j]
            total += peak - final
        return total / n

    @property
    def total_time(self) -> float:
        return self.feature_time + sum(self.fit_times)


# ── Feature extraction helpers ────────────────────────────────────────


def _build_pipeline(
    stages: list[dict],
    n_neurons: int,
    pool_levels: list[int] | None = None,
    pool_order: int = 1,
    standardize: bool = False,
    gcn: bool = False,
    lcn_kernel: int | None = None,
    parallel: bool = False,
    **nef_kwargs,
) -> ConvNEFPipeline:
    """Construct a ConvNEFPipeline without fitting it."""
    return ConvNEFPipeline(
        stages=stages,
        n_neurons=n_neurons,
        pool_levels=pool_levels,
        pool_order=pool_order,
        standardize=standardize,
        gcn=gcn,
        lcn_kernel=lcn_kernel,
        parallel=parallel,
        **nef_kwargs,
    )


def fit_pipeline_features(
    pipeline: ConvNEFPipeline,
    images: torch.Tensor,
    n_classes: int,
    *,
    fit_subsample: int = 10_000,
    batch_size: int = 1000,
    seed: int = 0,
) -> None:
    """Fit conv stages and create the NEF head (but do NOT solve decoders).

    After this call the pipeline can extract features and accumulate
    normal equations, but has no trained decoders yet.
    """
    N = images.shape[0]
    g = torch.Generator(device=images.device).manual_seed(seed)

    sub_n = min(fit_subsample, N)
    sub_idx = torch.randperm(N, generator=g, device=images.device)[:sub_n]
    sub_images = pipeline._preprocess(images[sub_idx])

    # Fit PCA stages
    if pipeline.parallel:
        for stage in pipeline.stages:
            stage.fit(sub_images)
    else:
        x_sub = sub_images
        for stage in pipeline.stages:
            stage.fit(x_sub)
            chunks = []
            for j in range(0, x_sub.shape[0], batch_size):
                chunks.append(stage(x_sub[j : j + batch_size]))
            x_sub = torch.cat(chunks, dim=0)
            del chunks

    # Extract features for standardization stats and centers
    center_chunks = []
    for i in range(0, sub_n, batch_size):
        chunk = pipeline._apply_stages(sub_images[i : i + batch_size])
        center_chunks.append(pipeline._pool_features(chunk))
        del chunk
    x_sub = torch.cat(center_chunks, dim=0)
    del center_chunks

    centers = pipeline._standardize_features(x_sub, fit=True)
    del x_sub
    feat_dim = centers.shape[1]

    # Create NEF head with data-driven biases
    head_kwargs = dict(pipeline.nef_kwargs)
    strategy = head_kwargs.get("encoder_strategy", "hypersphere")
    data_strategies = {"whitened", "class_contrast", "local_pca"}
    if strategy in data_strategies:
        ek = dict(head_kwargs.get("encoder_kwargs", {}) or {})
        if "train_data" not in ek:
            ek["train_data"] = centers
        head_kwargs["encoder_kwargs"] = ek

    pipeline.head = NEFLayer(
        feat_dim,
        pipeline.n_neurons,
        n_classes,
        centers=centers,
        **head_kwargs,
    )
    pipeline.head = pipeline.head.to(images.device)


@torch.no_grad()
def extract_features(
    pipeline: ConvNEFPipeline,
    images: torch.Tensor,
    batch_size: int = 1000,
) -> torch.Tensor:
    """Extract standardized features from a fitted pipeline."""
    chunks = []
    for i in range(0, images.shape[0], batch_size):
        x = pipeline._preprocess(images[i : i + batch_size])
        x = pipeline._apply_stages(x)
        x = pipeline._pool_features(x)
        x = pipeline._standardize_features(x)
        chunks.append(x)
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def evaluate_pipeline_on_tasks(
    pipeline: ConvNEFPipeline,
    tasks: list[dict],
    batch_size: int = 1000,
) -> list[float]:
    """Evaluate pipeline on all tasks, return per-task accuracy."""
    pipeline.eval()
    accs = []
    for task in tasks:
        pred = pipeline.predict(task["x_test"], batch_size=batch_size)
        acc = (pred.argmax(1) == task["y_test"]).float().mean().item()
        accs.append(acc)
    return accs


# ── CL experiments ────────────────────────────────────────────────────

N_CLASSES = 10


def run_convnef_cl(
    tasks: list[dict],
    feature_images: torch.Tensor,
    feature_mode: str,
    *,
    stages: list[dict],
    n_neurons: int = 10_000,
    alpha: float = 1e-2,
    pool_levels: list[int] | None = None,
    pool_order: int = 1,
    standardize: bool = False,
    gcn: bool = False,
    parallel: bool = False,
    fit_subsample: int = 10_000,
    batch_size: int = 1000,
    augment_flip: bool = False,
    n_augment: int = 1,
    seed: int = 0,
) -> ConvNEFCLResult:
    """Run ConvNEF + continual learning experiment.

    1. Fits pipeline features (PCA filters) on ``feature_images``.
    2. For each task: extracts features, accumulates via ``partial_fit``.
    3. After each task: solves decoders, evaluates on all tasks.
    """
    pipeline = _build_pipeline(
        stages=stages,
        n_neurons=n_neurons,
        pool_levels=pool_levels,
        pool_order=pool_order,
        standardize=standardize,
        gcn=gcn,
        parallel=parallel,
    )

    # Step 1: fit features
    t0 = time.perf_counter()
    fit_pipeline_features(
        pipeline,
        feature_images,
        N_CLASSES,
        fit_subsample=fit_subsample,
        batch_size=batch_size,
        seed=seed,
    )
    feature_time = time.perf_counter() - t0
    print(f"  Feature extraction ({feature_mode}): {feature_time:.1f}s", flush=True)

    # Step 2: continual decoding
    pipeline.head.reset_accumulators()
    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []

    augment_fn = None
    if augment_flip:
        augment_fn = lambda x: x.flip(-1)  # noqa: E731

    for task_idx, task in enumerate(tasks):
        targets = F.one_hot(task["y_train"], N_CLASSES).float()
        t0 = time.perf_counter()

        # Extract features and accumulate
        for i in range(0, task["x_train"].shape[0], batch_size):
            x_batch = task["x_train"][i : i + batch_size]
            t_batch = targets[i : i + batch_size]
            features = extract_features(pipeline, x_batch, batch_size=batch_size)
            pipeline.head.partial_fit(features, t_batch)
            del features

            if augment_fn is not None:
                for _ in range(n_augment):
                    x_aug = augment_fn(x_batch)
                    f_aug = extract_features(pipeline, x_aug, batch_size=batch_size)
                    pipeline.head.partial_fit(f_aug, t_batch)
                    del f_aug

        pipeline.head.solve_accumulated(alpha=alpha)
        fit_time = time.perf_counter() - t0
        fit_times.append(fit_time)

        accs = evaluate_pipeline_on_tasks(pipeline, tasks, batch_size=batch_size)
        accuracy_matrix.append(accs)

        seen_tasks = task_idx + 1
        avg_seen = sum(accs[:seen_tasks]) / seen_tasks
        print(
            f"  Task {task_idx + 1}/{len(tasks)} ({task['name']}): "
            f"avg(seen)={avg_seen:.1%}, fit={fit_time:.1f}s",
            flush=True,
        )

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
            "stages": stages,
            "pool_levels": pool_levels,
            "pool_order": pool_order,
            "standardize": standardize,
            "parallel": parallel,
            "augment_flip": augment_flip,
        },
    )


def run_convnef_joint(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    tasks: list[dict],
    *,
    stages: list[dict],
    n_neurons: int = 10_000,
    alpha: float = 1e-2,
    pool_levels: list[int] | None = None,
    pool_order: int = 1,
    standardize: bool = False,
    gcn: bool = False,
    parallel: bool = False,
    fit_subsample: int = 10_000,
    batch_size: int = 1000,
    augment_flip: bool = False,
    n_augment: int = 1,
    seed: int = 0,
) -> ConvNEFCLResult:
    """ConvNEF trained jointly on all data (non-CL upper bound)."""
    pipeline = _build_pipeline(
        stages=stages,
        n_neurons=n_neurons,
        pool_levels=pool_levels,
        pool_order=pool_order,
        standardize=standardize,
        gcn=gcn,
        parallel=parallel,
    )

    targets = F.one_hot(y_train, N_CLASSES).float()
    augment_fn = (lambda x: x.flip(-1)) if augment_flip else None

    t0 = time.perf_counter()
    pipeline.fit(
        x_train,
        targets,
        alpha=alpha,
        fit_subsample=fit_subsample,
        batch_size=batch_size,
        seed=seed,
        augment_fn=augment_fn,
        n_augment=n_augment,
    )
    total_time = time.perf_counter() - t0

    accs = evaluate_pipeline_on_tasks(pipeline, tasks, batch_size=batch_size)

    return ConvNEFCLResult(
        method="ConvNEF-joint (upper bound)",
        feature_mode="joint",
        n_neurons=n_neurons,
        n_tasks=len(tasks),
        accuracy_matrix=[accs],
        fit_times=[total_time],
        feature_time=0.0,
        config={
            "alpha": alpha,
            "stages": stages,
            "pool_levels": pool_levels,
            "pool_order": pool_order,
        },
    )


def run_flat_nef_cl(
    tasks: list[dict],
    *,
    n_neurons: int = 5000,
    alpha: float = 1e-2,
    seed: int = 0,
) -> ConvNEFCLResult:
    """Flat-pixel NEF accumulation baseline (no conv features)."""
    torch.manual_seed(seed)
    d_in = tasks[0]["x_train"].shape[1:].numel()
    device = tasks[0]["x_train"].device

    # Use first-task centers
    flat_t0 = tasks[0]["x_train"].reshape(-1, d_in)
    g = torch.Generator(device=device).manual_seed(seed)
    sub_n = min(10_000, flat_t0.shape[0])
    sub_idx = torch.randperm(flat_t0.shape[0], generator=g, device=device)[:sub_n]
    centers = flat_t0[sub_idx]

    layer = NEFLayer(
        d_in, n_neurons, N_CLASSES, activation="abs", gain=(0.5, 2.0), centers=centers
    )
    layer = layer.to(device)

    accuracy_matrix: list[list[float]] = []
    fit_times: list[float] = []

    for task_idx, task in enumerate(tasks):
        x_flat = task["x_train"].reshape(-1, d_in)
        targets = F.one_hot(task["y_train"], N_CLASSES).float()
        t0 = time.perf_counter()
        layer.partial_fit(x_flat, targets)
        layer.solve_accumulated(alpha=alpha)
        fit_times.append(time.perf_counter() - t0)

        accs = []
        for eval_task in tasks:
            x_eval = eval_task["x_test"].reshape(-1, d_in)
            with torch.no_grad():
                pred = layer(x_eval)
            acc = (pred.argmax(1) == eval_task["y_test"]).float().mean().item()
            accs.append(acc)
        accuracy_matrix.append(accs)

        seen_tasks = task_idx + 1
        avg_seen = sum(accs[:seen_tasks]) / seen_tasks
        print(
            f"  Task {task_idx + 1}/{len(tasks)} ({task['name']}): "
            f"avg(seen)={avg_seen:.1%}, fit={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    return ConvNEFCLResult(
        method="Flat NEF CL (baseline)",
        feature_mode="none",
        n_neurons=n_neurons,
        n_tasks=len(tasks),
        accuracy_matrix=accuracy_matrix,
        fit_times=fit_times,
        feature_time=0.0,
        config={"alpha": alpha},
    )


# ── Display ───────────────────────────────────────────────────────────


def print_result(result: ConvNEFCLResult) -> None:
    """Pretty-print a CL result with accuracy matrix and summary."""
    print(f"\n{'=' * 70}")
    print(f"  {result.method}")
    print(f"  {result.n_neurons} neurons, {result.n_tasks} tasks")
    print(f"{'=' * 70}")

    if len(result.accuracy_matrix) > 1:
        print("\nAccuracy matrix (rows = after task t, columns = task j):")
        header = "       " + "".join(f"  Task {j + 1}" for j in range(result.n_tasks))
        print(header)
        for i, row in enumerate(result.accuracy_matrix):
            vals = "".join(f"  {v:6.1%}" for v in row)
            print(f"  T{i + 1}:  {vals}")

    print(f"\n  Final avg accuracy: {result.final_avg_accuracy:.1%}")
    print(f"  Forgetting:         {result.forgetting:.1%}")
    print(f"  Feature time:       {result.feature_time:.1f}s")
    print(f"  Total time:         {result.total_time:.1f}s")


# ── Main ──────────────────────────────────────────────────────────────

# Good ConvNEF config from conv_cifar_v7 experiments
DEFAULT_STAGES = [
    {"n_filters": 32, "patch_size": 3, "pool_size": 1},
    {"n_filters": 32, "patch_size": 5, "pool_size": 1},
    {"n_filters": 32, "patch_size": 7, "pool_size": 1},
]


def main(argv: list[str] | None = None) -> list[ConvNEFCLResult]:
    parser = argparse.ArgumentParser(description="ConvNEF + continual learning benchmark")
    parser.add_argument("--n-neurons", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--fit-subsample", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--augment-flip", action="store_true", help="Horizontal flip augmentation")
    parser.add_argument("--quick", action="store_true", help="Smaller config for quick validation")
    args = parser.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    print(f"Device: {device}, neurons: {args.n_neurons}, seed: {args.seed}", flush=True)

    # Load data
    print("Loading CIFAR-10...", flush=True)
    x_train, y_train, x_test, y_test = load_cifar10_images(args.data_root, device)
    tasks = split_cifar10_tasks(x_train, y_train, x_test, y_test)
    print(f"Split-CIFAR-10: {len(tasks)} tasks", flush=True)

    stages = DEFAULT_STAGES
    n_neurons = args.n_neurons
    if args.quick:
        stages = [{"n_filters": 16, "patch_size": 5, "pool_size": 2}]
        n_neurons = min(n_neurons, 2000)

    conv_kwargs = dict(
        stages=stages,
        n_neurons=n_neurons,
        alpha=args.alpha,
        pool_levels=[1, 2, 4],
        pool_order=1,
        standardize=True,
        parallel=True,
        fit_subsample=args.fit_subsample,
        batch_size=args.batch_size,
        augment_flip=args.augment_flip,
        seed=args.seed,
    )

    results: list[ConvNEFCLResult] = []

    # 1. ConvNEF-CL with all-data features (oracle)
    print("\n[1/4] ConvNEF-CL (all_data features)...", flush=True)
    r = run_convnef_cl(tasks, x_train, "all_data", **conv_kwargs)
    print_result(r)
    results.append(r)

    # 2. ConvNEF-CL with first-task features
    print("\n[2/4] ConvNEF-CL (first_task features)...", flush=True)
    r = run_convnef_cl(tasks, tasks[0]["x_train"], "first_task", **conv_kwargs)
    print_result(r)
    results.append(r)

    # 3. ConvNEF-joint (non-CL upper bound)
    print("\n[3/4] ConvNEF-joint (non-CL)...", flush=True)
    r = run_convnef_joint(x_train, y_train, tasks, **conv_kwargs)
    print_result(r)
    results.append(r)

    # 4. Flat NEF CL baseline
    flat_neurons = min(n_neurons, 5000)
    print(f"\n[4/4] Flat NEF CL ({flat_neurons} neurons)...", flush=True)
    r = run_flat_nef_cl(tasks, n_neurons=flat_neurons, alpha=args.alpha, seed=args.seed)
    print_result(r)
    results.append(r)

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Method':<40} {'Avg Acc':>8} {'Forget':>8} {'Time':>8}")
    print(f"  {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 8}")
    for r in results:
        print(
            f"  {r.method:<40} {r.final_avg_accuracy:>7.1%} "
            f"{r.forgetting:>7.1%} {r.total_time:>7.1f}s"
        )

    # Save results
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "convnef_cl.json"
        serializable = []
        for r in results:
            d = asdict(r)
            d["final_avg_accuracy"] = r.final_avg_accuracy
            d["forgetting"] = r.forgetting
            serializable.append(d)
        out_path.write_text(json.dumps(serializable, indent=2))
        print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
