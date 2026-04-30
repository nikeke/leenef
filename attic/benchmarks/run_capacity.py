"""Capacity scaling experiments for NEF continual learning.

Measures how NEF accuracy scales with neuron count and task count,
and whether the joint-training equivalence holds at extremes.

Experiments:
  1. Permuted tasks: vary n_tasks × n_neurons, measure final accuracy
  2. Spot-check joint equivalence at extreme configurations

Supports MNIST and CIFAR-10 datasets.

Usage:
  python benchmarks/run_capacity.py --dataset mnist --seed 0
  python benchmarks/run_capacity.py --dataset cifar10 --max-tasks 20 --seed 0
  python benchmarks/run_capacity.py --max-tasks 100 --neuron-counts 500 1000 2000 5000 10000 --seed 0
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from leenef.layers import NEFLayer  # noqa: E402

# ── Utilities ─────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot(labels: Tensor, n_classes: int) -> Tensor:
    return torch.zeros(len(labels), n_classes).scatter_(1, labels.unsqueeze(1), 1.0)


def load_mnist(root: str = "./data"):
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    train = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)

    def to_tensors(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        x, y = next(iter(loader))
        return x.reshape(x.shape[0], -1).float(), y

    return to_tensors(train), to_tensors(test)


def load_cifar10(root: str = "./data"):
    import torchvision
    import torchvision.transforms as T

    transform = T.ToTensor()
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform)

    def to_tensors(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        x, y = next(iter(loader))
        return x.reshape(x.shape[0], -1).float(), y.long()

    return to_tensors(train), to_tensors(test)


DATASET_LOADERS = {
    "mnist": load_mnist,
    "cifar10": load_cifar10,
}


def make_permuted_tasks(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    n_tasks: int,
    seed: int,
) -> list[dict]:
    """Create permuted-pixel tasks."""
    tasks = [
        {
            "x_train": x_train.clone(),
            "y_train": y_train.clone(),
            "x_test": x_test.clone(),
            "y_test": y_test.clone(),
        }
    ]
    rng = torch.Generator().manual_seed(seed + 1000)
    for _ in range(1, n_tasks):
        perm = torch.randperm(x_train.shape[1], generator=rng)
        tasks.append(
            {
                "x_train": x_train[:, perm],
                "y_train": y_train.clone(),
                "x_test": x_test[:, perm],
                "y_test": y_test.clone(),
            }
        )
    return tasks


# ── Core experiments ──────────────────────────────────────────────────


def run_accumulate(
    tasks: list[dict],
    n_neurons: int,
    n_classes: int,
    alpha: float,
    seed: int,
) -> dict:
    """Run NEF-accumulate on tasks, return per-task accuracy after all tasks."""
    set_seed(seed)
    d_in = tasks[0]["x_train"].shape[1]
    layer = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))

    t0 = time.perf_counter()
    for task in tasks:
        targets = one_hot(task["y_train"], n_classes)
        layer.partial_fit(task["x_train"], targets)
    layer.solve_accumulated(alpha=alpha)
    elapsed = time.perf_counter() - t0

    # Evaluate on all tasks
    layer.eval()
    per_task_acc = []
    with torch.no_grad():
        for task in tasks:
            pred = layer(task["x_test"])
            acc = (pred.argmax(1) == task["y_test"]).float().mean().item()
            per_task_acc.append(acc)

    return {
        "avg_acc": sum(per_task_acc) / len(per_task_acc),
        "min_acc": min(per_task_acc),
        "max_acc": max(per_task_acc),
        "std_acc": float(np.std(per_task_acc)),
        "time": elapsed,
        "per_task_acc": per_task_acc,
    }


def run_joint(
    tasks: list[dict],
    n_neurons: int,
    n_classes: int,
    alpha: float,
    seed: int,
) -> dict:
    """Run NEF-joint using batched partial_fit (memory-efficient)."""
    set_seed(seed)
    d_in = tasks[0]["x_train"].shape[1]
    layer = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))

    t0 = time.perf_counter()
    for task in tasks:
        targets = one_hot(task["y_train"], n_classes)
        layer.partial_fit(task["x_train"], targets)
    layer.solve_accumulated(alpha=alpha)
    elapsed = time.perf_counter() - t0

    layer.eval()
    per_task_acc = []
    with torch.no_grad():
        for task in tasks:
            pred = layer(task["x_test"])
            acc = (pred.argmax(1) == task["y_test"]).float().mean().item()
            per_task_acc.append(acc)

    return {
        "avg_acc": sum(per_task_acc) / len(per_task_acc),
        "time": elapsed,
    }


# ── Main sweep ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="NEF capacity scaling experiments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="results/continual")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=50,
        help="Maximum number of permuted tasks (default: 50)",
    )
    parser.add_argument(
        "--neuron-counts",
        type=int,
        nargs="+",
        default=None,
        help="Neuron counts to sweep (default: dataset-dependent)",
    )
    parser.add_argument(
        "--task-counts",
        type=int,
        nargs="+",
        default=None,
        help="Task counts to sweep (default: 5,10,20,50 up to max-tasks)",
    )
    args = parser.parse_args()

    default_tasks = [t for t in [5, 10, 20, 50, 100] if t <= args.max_tasks]
    task_counts = args.task_counts or default_tasks
    if args.neuron_counts is not None:
        neuron_counts = args.neuron_counts
    elif args.dataset == "cifar10":
        neuron_counts = [1000, 2000, 5000, 10000]
    else:
        neuron_counts = [500, 1000, 2000, 5000]

    ds_name = args.dataset.upper()
    print(f"Loading {ds_name} ...")
    loader = DATASET_LOADERS[args.dataset]
    (x_train, y_train), (x_test, y_test) = loader(args.data_root)
    n_classes = int(y_train.max().item()) + 1
    print(f"  train: {x_train.shape}, test: {x_test.shape}, {n_classes} classes")

    # Pre-generate all permuted tasks up to max needed
    max_t = max(task_counts)
    print(f"Generating {max_t} permuted tasks ...")
    all_tasks = make_permuted_tasks(x_train, y_train, x_test, y_test, max_t, args.seed)

    # Measure single-task accuracy for relative scaling
    print("\nSingle-task accuracy (baseline):")
    single_task_accs = {}
    for n_neurons in neuron_counts:
        r = run_accumulate(all_tasks[:1], n_neurons, n_classes, args.alpha, args.seed)
        single_task_accs[n_neurons] = r["avg_acc"]
        print(f"  {n_neurons:5d} neurons: {r['avg_acc'] * 100:.1f}%")

    print(f"\nCapacity sweep: {len(task_counts)} task counts × {len(neuron_counts)} neuron counts")
    print(f"Task counts:   {task_counts}")
    print(f"Neuron counts: {neuron_counts}")

    # ── Run sweep ──
    results = {}
    for n_neurons in neuron_counts:
        results[n_neurons] = {}
        for n_tasks in task_counts:
            tasks = all_tasks[:n_tasks]
            label = f"neurons={n_neurons}, tasks={n_tasks}"
            print(f"\n  Running {label} ...", end="", flush=True)
            r = run_accumulate(tasks, n_neurons, n_classes, args.alpha, args.seed)
            results[n_neurons][n_tasks] = r
            print(
                f"  avg={r['avg_acc'] * 100:.1f}%  "
                f"min={r['min_acc'] * 100:.1f}%  "
                f"std={r['std_acc'] * 100:.1f}%  "
                f"time={r['time']:.1f}s"
            )

    # ── Joint equivalence spot-checks ──
    print("\n\nJoint-training equivalence spot-checks:")
    print("-" * 60)
    # Check the smallest and largest configs
    spot_checks = [
        (neuron_counts[0], task_counts[-1]),
        (neuron_counts[-1], task_counts[-1]),
    ]
    for n_neurons, n_tasks in spot_checks:
        tasks = all_tasks[:n_tasks]
        joint = run_joint(tasks, n_neurons, n_classes, args.alpha, args.seed)
        accum = results[n_neurons][n_tasks]
        gap = abs(accum["avg_acc"] - joint["avg_acc"]) * 100
        print(
            f"  neurons={n_neurons:5d}, tasks={n_tasks:3d}: "
            f"accum={accum['avg_acc'] * 100:.2f}%  "
            f"joint={joint['avg_acc'] * 100:.2f}%  "
            f"gap={gap:.4f}%"
        )

    # ── Print capacity matrix ──
    print("\n\n" + "=" * 72)
    print(f"  CAPACITY MATRIX: Permuted-{ds_name} Final Avg Accuracy (%)")
    print("=" * 72)

    # Header
    hdr = f"{'Tasks':>8s} |"
    for n in neuron_counts:
        hdr += f" {n:>7d} |"
    print(hdr)
    print("-" * len(hdr))

    for n_tasks in task_counts:
        row = f"{n_tasks:>8d} |"
        for n_neurons in neuron_counts:
            acc = results[n_neurons][n_tasks]["avg_acc"] * 100
            row += f" {acc:6.1f}% |"
        print(row)

    # ── Timing matrix ──
    print("\n  TIMING MATRIX: Total fit time (seconds)")
    print("-" * (10 + 10 * len(neuron_counts)))
    hdr = f"{'Tasks':>8s} |"
    for n in neuron_counts:
        hdr += f" {n:>7d} |"
    print(hdr)
    print("-" * len(hdr))
    for n_tasks in task_counts:
        row = f"{n_tasks:>8d} |"
        for n_neurons in neuron_counts:
            t = results[n_neurons][n_tasks]["time"]
            row += f" {t:6.1f}s |"
        print(row)

    # ── Accuracy std matrix (cross-task consistency) ──
    print("\n  CROSS-TASK STD MATRIX: Std of per-task accuracy (%)")
    print("-" * (10 + 10 * len(neuron_counts)))
    hdr = f"{'Tasks':>8s} |"
    for n in neuron_counts:
        hdr += f" {n:>7d} |"
    print(hdr)
    print("-" * len(hdr))
    for n_tasks in task_counts:
        row = f"{n_tasks:>8d} |"
        for n_neurons in neuron_counts:
            s = results[n_neurons][n_tasks]["std_acc"] * 100
            row += f"  {s:5.2f}% |"
        print(row)

    # ── Memory footprint ──
    print("\n  MEMORY FOOTPRINT: AᵀA matrix size")
    print("-" * 50)
    for n_neurons in neuron_counts:
        mem_mb = n_neurons * n_neurons * 8 / (1024 * 1024)  # float64
        print(f"  {n_neurons:5d} neurons: AᵀA = {n_neurons}×{n_neurons} = {mem_mb:.1f} MB")
    print("  (AᵀA size is constant regardless of number of tasks)")

    # ── Relative capacity matrix (% of single-task accuracy retained) ──
    print("\n  RELATIVE CAPACITY: % of single-task accuracy retained")
    print("-" * (10 + 10 * len(neuron_counts)))
    hdr = f"{'Tasks':>8s} |"
    for n in neuron_counts:
        hdr += f" {n:>7d} |"
    print(hdr)
    print("-" * len(hdr))
    for n_tasks in task_counts:
        row = f"{n_tasks:>8d} |"
        for n_neurons in neuron_counts:
            acc = results[n_neurons][n_tasks]["avg_acc"]
            baseline = single_task_accs[n_neurons]
            rel = (acc / baseline * 100) if baseline > 0 else 0
            row += f" {rel:6.1f}% |"
        print(row)

    # ── Neurons per output dimension analysis ──
    print("\n  NEURONS PER OUTPUT DIMENSION: n / (tasks × classes)")
    print("-" * (10 + 10 * len(neuron_counts)))
    hdr = f"{'Tasks':>8s} |"
    for n in neuron_counts:
        hdr += f" {n:>7d} |"
    print(hdr)
    print("-" * len(hdr))
    for n_tasks in task_counts:
        row = f"{n_tasks:>8d} |"
        for n_neurons in neuron_counts:
            ratio = n_neurons / (n_tasks * n_classes)
            row += f" {ratio:7.1f} |"
        print(row)

    # ── Save results ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"capacity_{args.dataset}_results.json"

    payload = {
        "experiment": f"permuted_{args.dataset}_capacity",
        "dataset": args.dataset,
        "seed": args.seed,
        "alpha": args.alpha,
        "neuron_counts": neuron_counts,
        "task_counts": task_counts,
        "single_task_accs": {str(n): a for n, a in single_task_accs.items()},
        "results": {
            str(n): {
                str(t): {k: v for k, v in r.items() if k != "per_task_acc"} for t, r in tr.items()
            }
            for n, tr in results.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
