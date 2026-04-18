"""Woodbury vs accumulate comparison for continual learning.

Investigates two questions:
  Q1: Does continuous_fit (fixed α, Woodbury updates) produce the same
      results as partial_fit + solve_accumulated (trace-scaled α)?
  Q2: Can Woodbury enable online (mini-batch) continual learning, and
      how does numerical drift affect accuracy?

Usage:
  python benchmarks/run_woodbury.py --seed 0
  python benchmarks/run_woodbury.py --dataset cifar10 --seed 0
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from run_continual import (  # noqa: E402, I001
    DATASET_DEFAULT_NEURONS,
    DATASET_LOADERS,
    DATASET_N_CLASSES,
    one_hot,
    permuted_tasks,
    split_class_tasks,
)

from leenef.layers import NEFLayer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate_on_tasks(model, tasks: list[dict]) -> list[float]:
    model.eval()
    accs = []
    for task in tasks:
        pred = model(task["x_test"])
        acc = (pred.argmax(dim=1) == task["y_test"]).float().mean().item()
        accs.append(acc)
    return accs


# ── Comparison runners ────────────────────────────────────────────────


@dataclass
class WoodburyResult:
    method: str
    scenario: str
    n_tasks: int
    n_neurons: int
    accuracy_matrix: list[list[float]]
    final_avg_acc: float
    forgetting: float
    total_time: float
    batch_size: int | None = None
    decoder_norm_diff: list[float] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def run_accumulate(
    tasks: list[dict], scenario: str, *, n_neurons: int, alpha: float, seed: int
) -> WoodburyResult:
    """Baseline: partial_fit + solve_accumulated (trace-scaled α)."""
    set_seed(seed)
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1
    d_in = tasks[0]["x_train"].shape[1]
    layer = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))

    acc_matrix: list[list[float]] = []
    t0 = time.perf_counter()
    for task in tasks:
        targets = one_hot(task["y_train"], n_classes)
        layer.partial_fit(task["x_train"], targets)
        layer.solve_accumulated(alpha=alpha)
        acc_matrix.append(evaluate_on_tasks(layer, tasks))
    total = time.perf_counter() - t0

    final_accs = acc_matrix[-1]
    avg = sum(final_accs) / len(final_accs)
    forg = _forgetting(acc_matrix)

    return WoodburyResult(
        method="accumulate (trace-scaled α)",
        scenario=scenario,
        n_tasks=len(tasks),
        n_neurons=n_neurons,
        accuracy_matrix=acc_matrix,
        final_avg_acc=avg,
        forgetting=forg,
        total_time=total,
        config={"alpha": alpha, "regularization": "trace-scaled"},
    )


def run_woodbury_batch(
    tasks: list[dict], scenario: str, *, n_neurons: int, alpha: float, seed: int
) -> WoodburyResult:
    """Woodbury with per-task batches via continuous_fit (fixed α)."""
    set_seed(seed)
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1
    d_in = tasks[0]["x_train"].shape[1]
    layer = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))

    acc_matrix: list[list[float]] = []
    t0 = time.perf_counter()
    for task in tasks:
        targets = one_hot(task["y_train"], n_classes)
        layer.continuous_fit(task["x_train"], targets, alpha=alpha)
        acc_matrix.append(evaluate_on_tasks(layer, tasks))
    total = time.perf_counter() - t0

    final_accs = acc_matrix[-1]
    avg = sum(final_accs) / len(final_accs)
    forg = _forgetting(acc_matrix)

    return WoodburyResult(
        method="woodbury-batch (fixed α)",
        scenario=scenario,
        n_tasks=len(tasks),
        n_neurons=n_neurons,
        accuracy_matrix=acc_matrix,
        final_avg_acc=avg,
        forgetting=forg,
        total_time=total,
        config={"alpha": alpha, "regularization": "fixed"},
    )


def run_woodbury_online(
    tasks: list[dict],
    scenario: str,
    *,
    n_neurons: int,
    alpha: float,
    seed: int,
    batch_size: int = 100,
) -> WoodburyResult:
    """Woodbury with mini-batch updates — real rank-k Woodbury updates.

    Splits each task's data into mini-batches of ``batch_size`` and calls
    continuous_fit for each.  Evaluates after each full task.
    """
    set_seed(seed)
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1
    d_in = tasks[0]["x_train"].shape[1]
    layer = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))

    acc_matrix: list[list[float]] = []
    t0 = time.perf_counter()
    for task in tasks:
        x = task["x_train"]
        targets = one_hot(task["y_train"], n_classes)
        n = x.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            layer.continuous_fit(x[start:end], targets[start:end], alpha=alpha)
        acc_matrix.append(evaluate_on_tasks(layer, tasks))
    total = time.perf_counter() - t0

    final_accs = acc_matrix[-1]
    avg = sum(final_accs) / len(final_accs)
    forg = _forgetting(acc_matrix)

    return WoodburyResult(
        method=f"woodbury-online (bs={batch_size})",
        scenario=scenario,
        n_tasks=len(tasks),
        n_neurons=n_neurons,
        accuracy_matrix=acc_matrix,
        final_avg_acc=avg,
        forgetting=forg,
        total_time=total,
        batch_size=batch_size,
        config={"alpha": alpha, "batch_size": batch_size, "regularization": "fixed"},
    )


def run_drift_analysis(
    tasks: list[dict],
    scenario: str,
    *,
    n_neurons: int,
    alpha: float,
    seed: int,
    batch_size: int = 100,
) -> dict:
    """Measure numerical drift: compare Woodbury decoders vs exact re-solve.

    After processing each task's mini-batches via Woodbury, compares:
    - Woodbury decoders (accumulated rank-k updates)
    - Exact decoders from refresh_inverse (same α, full re-solve from AᵀA)
    - Accumulate-path decoders (trace-scaled α, different regularization)

    Reports decoder difference norms and accuracy differences.
    """
    set_seed(seed)
    n_classes = max(int(t["y_train"].max().item()) for t in tasks) + 1
    d_in = tasks[0]["x_train"].shape[1]

    # Woodbury layer (online mini-batches)
    layer_wb = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))
    # Accumulate layer (for comparison)
    set_seed(seed)
    layer_acc = NEFLayer(d_in, n_neurons, n_classes, activation="abs", gain=(0.5, 2.0))

    drift_log: list[dict] = []
    total_batches = 0

    for ti, task in enumerate(tasks):
        x = task["x_train"]
        targets = one_hot(task["y_train"], n_classes)
        n = x.shape[0]

        # Woodbury: mini-batch updates
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            layer_wb.continuous_fit(x[start:end], targets[start:end], alpha=alpha)
            total_batches += 1

        # Accumulate: one-shot partial_fit + solve
        layer_acc.partial_fit(x, targets)
        layer_acc.solve_accumulated(alpha=alpha)

        # Snapshot Woodbury decoders and state before refresh
        D_woodbury = layer_wb.decoders.data.clone()
        M_inv_saved = layer_wb._M_inv.clone()
        ata_saved = layer_wb._ata.clone()

        # Refresh inverse from accumulated AᵀA (same fixed α).
        # NOTE: refresh_inverse mutates _ata (adds α to diagonal) and
        # overwrites _M_inv, so we must restore both afterward.
        layer_wb.refresh_inverse(alpha=alpha)
        D_refreshed = layer_wb.decoders.data.clone()

        # Restore Woodbury state so subsequent tasks continue correctly
        layer_wb._M_inv = M_inv_saved
        layer_wb._ata = ata_saved
        layer_wb.decoders.data.copy_(D_woodbury)

        # Accumulate decoders (trace-scaled α)
        D_accumulate = layer_acc.decoders.data.clone()

        # Norms
        wb_vs_refresh = (D_woodbury - D_refreshed).norm().item()
        wb_vs_accum = (D_woodbury - D_accumulate).norm().item()
        refresh_vs_accum = (D_refreshed - D_accumulate).norm().item()

        # Accuracy comparison
        acc_wb_pre = evaluate_on_tasks(layer_wb, tasks)

        # Temporarily set refreshed decoders for evaluation
        layer_wb.decoders.data.copy_(D_refreshed)
        acc_wb = evaluate_on_tasks(layer_wb, tasks)
        # Restore Woodbury decoders
        layer_wb.decoders.data.copy_(D_woodbury)

        acc_acc = evaluate_on_tasks(layer_acc, tasks)

        avg_wb_pre = sum(acc_wb_pre) / len(acc_wb_pre)
        avg_wb_refresh = sum(acc_wb) / len(acc_wb)
        avg_acc = sum(acc_acc) / len(acc_acc)

        entry = {
            "task": ti,
            "total_batches": total_batches,
            "decoder_norm_woodbury": D_woodbury.norm().item(),
            "drift_wb_vs_refresh": wb_vs_refresh,
            "drift_wb_vs_accum": wb_vs_accum,
            "drift_refresh_vs_accum": refresh_vs_accum,
            "acc_woodbury": avg_wb_pre,
            "acc_refreshed": avg_wb_refresh,
            "acc_accumulate": avg_acc,
            "acc_diff_drift": abs(avg_wb_pre - avg_wb_refresh),
            "acc_diff_reg": abs(avg_wb_refresh - avg_acc),
        }
        drift_log.append(entry)

    return {
        "scenario": scenario,
        "n_neurons": n_neurons,
        "alpha": alpha,
        "batch_size": batch_size,
        "n_tasks": len(tasks),
        "drift_log": drift_log,
    }


def _forgetting(acc_matrix: list[list[float]]) -> float:
    n_rows = len(acc_matrix)
    if n_rows <= 1:
        return 0.0
    n = len(acc_matrix[-1])
    total = 0.0
    for j in range(n):
        peak = max(acc_matrix[i][j] for i in range(n_rows))
        final = acc_matrix[-1][j]
        total += peak - final
    return total / n


# ── Display ───────────────────────────────────────────────────────────


def print_comparison(results: list[WoodburyResult]) -> None:
    print(f"\n{'Method':<35s} {'Avg Acc':>8s} {'Forget':>8s} {'Time':>8s}")
    print("-" * 65)
    for r in results:
        print(
            f"{r.method:<35s} "
            f"{r.final_avg_acc * 100:7.2f}% "
            f"{r.forgetting * 100:7.2f}% "
            f"{r.total_time:7.1f}s"
        )


def print_drift(drift: dict) -> None:
    print(f"\n  Drift analysis (batch_size={drift['batch_size']}):")
    print(
        f"  {'Task':>4s} | {'Batches':>7s} | {'WB-Refresh':>10s} | "
        f"{'WB-Accum':>10s} | {'Ref-Accum':>10s} | "
        f"{'Acc(WB)':>8s} {'Acc(Ref)':>8s} {'Acc(Acc)':>8s}"
    )
    print("  " + "-" * 95)
    for e in drift["drift_log"]:
        print(
            f"  {e['task']:4d} | {e['total_batches']:7d} | "
            f"{e['drift_wb_vs_refresh']:10.2e} | "
            f"{e['drift_wb_vs_accum']:10.2e} | "
            f"{e['drift_refresh_vs_accum']:10.2e} | "
            f"{e['acc_woodbury'] * 100:7.2f}% "
            f"{e['acc_refreshed'] * 100:7.2f}% "
            f"{e['acc_accumulate'] * 100:7.2f}%"
        )


# ── Main ──────────────────────────────────────────────────────────────


def run_scenario(
    scenario_name: str,
    tasks: list[dict],
    *,
    n_neurons: int,
    alpha: float,
    seed: int,
    online_batch_sizes: list[int],
) -> tuple[list[WoodburyResult], list[dict]]:
    """Run all Woodbury comparisons for one scenario."""
    results: list[WoodburyResult] = []
    drift_results: list[dict] = []

    # 1. Accumulate baseline
    print("\n  [1/n] accumulate (trace-scaled α) ...")
    r = run_accumulate(tasks, scenario_name, n_neurons=n_neurons, alpha=alpha, seed=seed)
    results.append(r)

    # 2. Woodbury batch (per-task, fixed α)
    print("  [2/n] woodbury-batch (fixed α) ...")
    r = run_woodbury_batch(tasks, scenario_name, n_neurons=n_neurons, alpha=alpha, seed=seed)
    results.append(r)

    # 3. Woodbury online at various batch sizes
    for i, bs in enumerate(online_batch_sizes):
        print(f"  [{3 + i}/n] woodbury-online (bs={bs}) ...")
        r = run_woodbury_online(
            tasks, scenario_name, n_neurons=n_neurons, alpha=alpha, seed=seed, batch_size=bs
        )
        results.append(r)

    # 4. Drift analysis at smallest batch size
    bs_drift = min(online_batch_sizes)
    print(f"  [drift] numerical drift analysis (bs={bs_drift}) ...")
    drift = run_drift_analysis(
        tasks, scenario_name, n_neurons=n_neurons, alpha=alpha, seed=seed, batch_size=bs_drift
    )
    drift_results.append(drift)

    print_comparison(results)
    for d in drift_results:
        print_drift(d)

    return results, drift_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Woodbury vs accumulate comparison")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_LOADERS),
        default="mnist",
        help="Dataset (default: mnist)",
    )
    parser.add_argument(
        "--scenario", choices=["split", "permuted", "both"], default="both", help="CL scenario"
    )
    parser.add_argument("--n-neurons", type=int, default=None, help="NEF neurons")
    parser.add_argument("--alpha", type=float, default=1e-2, help="Tikhonov alpha")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--output-dir", type=str, default="results/continual")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--online-batch-sizes",
        type=int,
        nargs="+",
        default=[500, 100, 10],
        help="Mini-batch sizes for online Woodbury (default: 500 100 10)",
    )
    args = parser.parse_args()

    dataset = args.dataset
    n_classes = DATASET_N_CLASSES[dataset]
    n_neurons = args.n_neurons or DATASET_DEFAULT_NEURONS[dataset]

    print(f"Loading {dataset.upper()} ...")
    (x_train, y_train), (x_test, y_test) = DATASET_LOADERS[dataset](args.data_root)
    print(f"  train: {x_train.shape}, test: {x_test.shape}, {n_classes} classes")
    print(f"  neurons: {n_neurons}, alpha: {args.alpha}")

    all_results: list[WoodburyResult] = []
    all_drift: list[dict] = []

    if args.scenario in ("split", "both"):
        label = f"SPLIT-{dataset.upper()}"
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")
        tasks = split_class_tasks(x_train, y_train, x_test, y_test, n_classes, 2)
        results, drift = run_scenario(
            f"split-{dataset}",
            tasks,
            n_neurons=n_neurons,
            alpha=args.alpha,
            seed=args.seed,
            online_batch_sizes=args.online_batch_sizes,
        )
        all_results.extend(results)
        all_drift.extend(drift)

    if args.scenario in ("permuted", "both"):
        label = f"PERMUTED-{dataset.upper()}"
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")
        tasks = permuted_tasks(x_train, y_train, x_test, y_test, n_tasks=5, seed=args.seed)
        results, drift = run_scenario(
            f"permuted-{dataset}",
            tasks,
            n_neurons=n_neurons,
            alpha=args.alpha,
            seed=args.seed,
            online_batch_sizes=args.online_batch_sizes,
        )
        all_results.extend(results)
        all_drift.extend(drift)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_tag = args.scenario if args.scenario != "both" else "all"
    out_file = out_dir / f"woodbury_{dataset}_{scenario_tag}_results.json"
    payload = {
        "results": [
            {
                "method": r.method,
                "scenario": r.scenario,
                "n_tasks": r.n_tasks,
                "n_neurons": r.n_neurons,
                "final_avg_acc": r.final_avg_acc,
                "forgetting": r.forgetting,
                "total_time": r.total_time,
                "batch_size": r.batch_size,
                "config": r.config,
            }
            for r in all_results
        ],
        "drift": all_drift,
    }
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
