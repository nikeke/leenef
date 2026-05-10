"""Generate figures for `docs/SSUP-II.md`.

Reads `results/cl_clean/*.json` and writes PNGs to `docs/figures/`.
Idempotent: regenerate any single figure with `--only F2` etc.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "cl_clean"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


def _se(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return stdev(xs) / math.sqrt(len(xs))


def _load_rows(path: Path, *, clean_only: bool = True) -> list[dict]:
    data = json.loads(path.read_text())
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    if clean_only:
        rows = [r for r in rows if "-dirty-" not in r.get("code_identity", "")]
    counts = Counter(r.get("code_identity", "") for r in rows if r.get("code_identity"))
    if not counts:
        return rows
    identity, _ = counts.most_common(1)[0]
    return [r for r in rows if r.get("code_identity") == identity]


def _metric(r: dict, key: str = "test_accuracy") -> float:
    return float(r.get("metrics", {}).get(key, r.get(key, 0.0)))


def _time(r: dict) -> float:
    return float(r.get("metrics", {}).get("fit_time_s", r.get("wall_time_s", 0.0)))


def _group(rows: list[dict], key_fn) -> dict[str | int | tuple, list[dict]]:
    out: dict[str | int | tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[key_fn(r)].append(r)
    return out


# ------------------------------------------------------------------ F1
def figure_f1() -> Path:
    """Headline final accuracy across the four main CL panels."""

    panels = [
        ("Split-MNIST", "nef_continual_split_mnist_panel.json", (0, 100)),
        ("Split-CIFAR-10", "nef_continual_split_cifar10_panel.json", (0, 60)),
        ("Split-CIFAR-100", "nef_continual_split_cifar100_panel.json", (0, 30)),
        ("Permuted-MNIST", "nef_continual_permuted_mnist_panel.json", (0, 100)),
    ]
    cells = [
        ("nef_accumulate", "NEF accumulate", "C0"),
        ("mlp_finetune", "MLP finetune", "C3"),
        ("mlp_ewc", "MLP EWC", "C4"),
        ("mlp_er", "MLP ER(500)", "C2"),
        ("mlp_joint", "MLP joint", "C7"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    for ax, (title, file_name, ylim) in zip(axes, panels):
        rows = _load_rows(RESULTS / file_name)
        grouped = _group(rows, lambda r: r.get("config", {}).get("cell", str(r.get("value"))))
        xs = list(range(len(cells)))
        vals, ses, colors = [], [], []
        for cell, _, color in cells:
            accs = [100 * _metric(r) for r in grouped.get(cell, [])]
            vals.append(mean(accs) if accs else 0.0)
            ses.append(_se(accs) if accs else 0.0)
            colors.append(color)
        ax.bar(xs, vals, yerr=ses, capsize=3, color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label, _ in cells], rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("final accuracy (%)")
    fig.suptitle("Figure 1 -- Continual-learning final accuracy (10 seeds)")
    out = FIG_DIR / "cl_f1_headline_accuracy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F2
def figure_f2() -> Path:
    """Permuted-MNIST capacity table as a heat map."""

    task_counts = [5, 10, 20, 50, 100]
    n_values = [500, 1000, 2000, 5000, 10000]
    grid: list[list[float]] = []
    labels: list[list[str]] = []
    for n_tasks in task_counts:
        rows = _load_rows(RESULTS / f"b3_capacity_permuted_mnist_t{n_tasks}.json")
        grouped = _group(rows, lambda r: int(r.get("config", {}).get("n_neurons")))
        row_vals, row_labels = [], []
        for n in n_values:
            accs = [100 * _metric(r) for r in grouped[n]]
            m = mean(accs)
            row_vals.append(m)
            row_labels.append(f"{m:.1f}")
        grid.append(row_vals)
        labels.append(row_labels)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    image = ax.imshow(grid, cmap="viridis", vmin=50, vmax=95, aspect="auto")
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_yticks(range(len(task_counts)))
    ax.set_yticklabels([str(t) for t in task_counts])
    ax.set_xlabel("n_neurons")
    ax.set_ylabel("number of tasks")
    ax.set_title("Permuted-MNIST final accuracy")
    for i, row in enumerate(labels):
        for j, label in enumerate(row):
            color = "white" if grid[i][j] < 75 else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("final accuracy (%)")
    out = FIG_DIR / "cl_f2_capacity_table_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F3
def figure_f3() -> Path:
    """Solver mode accuracy/time trade-off at tuned alpha."""

    files = [
        ("Split-MNIST", "b5_solvers_split_mnist.json"),
        ("Split-CIFAR-10", "b5_solvers_split_cifar10.json"),
        ("Permuted-MNIST", "b5_solvers_permuted_mnist.json"),
    ]
    solvers = ["accumulate", "woodbury_batch", "woodbury_online"]
    colors = {"accumulate": "C0", "woodbury_batch": "C1", "woodbury_online": "C2"}

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharey=False)
    for ax, (title, file_name) in zip(axes, files):
        rows = _load_rows(RESULTS / file_name)
        best: dict[str, tuple[float, float, float]] = {}
        by_solver_alpha = _group(
            rows, lambda r: str(r.get("config", {}).get("solver_alpha_cell", ""))
        )
        by_solver: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for cell, group in by_solver_alpha.items():
            solver, _, alpha = cell.partition("_a")
            if solver not in solvers or not alpha:
                continue
            accs = [100 * _metric(r) for r in group]
            times = [_time(r) for r in group]
            by_solver[solver].append((mean(accs), _se(accs), mean(times)))
        for solver, vals in by_solver.items():
            best[solver] = max(vals, key=lambda x: x[0])

        for solver in solvers:
            acc, se, wall = best[solver]
            ax.errorbar(wall, acc, yerr=se, fmt="o", capsize=3, color=colors[solver], label=solver)
            ax.text(wall * 1.04, acc, solver.replace("_", "\n"), fontsize=8, va="center")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("fit time (s, log)")
        ax.set_ylabel("final accuracy (%)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Figure 3 -- Solver modes: same accuracy, different wall-clock")
    out = FIG_DIR / "cl_f3_solver_tradeoff.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F4
def figure_f4() -> Path:
    """Flat NEF vs ConvNEF on Split-CIFAR-10."""

    flat = _load_rows(RESULTS / "gpu_c2_flat_nef_split_cifar10_panel.json")
    conv = _load_rows(RESULTS / "gpu_c1_convnef_split_cifar10_panel.json")
    cells: list[tuple[str, list[dict]]] = [("flat NEF\n5000", flat)]
    grouped = _group(conv, lambda r: str(r.get("config", {}).get("cell", r.get("value"))))
    for cell in [
        "first_task_n10000",
        "all_data_n10000",
        "joint_n10000",
        "first_task_n20000",
        "all_data_n20000",
        "joint_n20000",
    ]:
        label = cell.replace("_n", "\n").replace("_", " ")
        cells.append((label, grouped[cell]))

    vals = [mean([100 * _metric(r) for r in rows]) for _, rows in cells]
    ses = [_se([100 * _metric(r) for r in rows]) for _, rows in cells]
    colors = ["C7"] + ["C0", "C1", "C2", "C0", "C1", "C2"]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    xs = list(range(len(cells)))
    ax.bar(xs, vals, yerr=ses, capsize=3, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _ in cells], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("final accuracy (%)")
    ax.set_ylim(45, 78)
    ax.set_title("Figure 4 -- ConvNEF raises the Split-CIFAR-10 ceiling")
    ax.grid(True, axis="y", alpha=0.3)
    for x, val in zip(xs, vals):
        ax.text(x, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    out = FIG_DIR / "cl_f4_convnef_lift.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F5
def figure_f5() -> Path:
    """Streaming continual follow-up on permuted sMNIST."""

    rows = _load_rows(RESULTS / "gpu_h8_streaming_permuted_smnist_panel.json")
    diag_rows = _load_rows(RESULTS / "gpu_h8_streaming_diagnostics.json")
    grouped = _group(rows, lambda r: str(r.get("config", {}).get("cell", r.get("value"))))
    diag_grouped = _group(
        diag_rows, lambda r: str(r.get("config", {}).get("cell", r.get("value")))
    )
    series = [
        ("accumulate_final", "accumulate = joint", "C0", "o"),
        ("woodbury_online", "Woodbury online", "C2", "s"),
        ("reset", "reset lower bound", "C3", "^"),
    ]
    n_values = [4000, 8000, 16000]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for prefix, label, color, marker in series:
        xs, vals, ses = [], [], []
        for n in n_values:
            cell = f"{prefix}_n{n}"
            if cell not in grouped:
                continue
            accs = [100 * _metric(r) for r in grouped[cell]]
            xs.append(n)
            vals.append(mean(accs))
            ses.append(_se(accs))
        if xs:
            ax.errorbar(xs, vals, yerr=ses, marker=marker, capsize=3, color=color, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("streaming neurons")
    ax.set_ylabel("final accuracy across 10 permutations (%)")
    ax.set_title("Width sweep (5 seeds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    probe_cells = [
        ("refresh_n4000_a1e-3_t10", "Woodbury\nbefore refresh", "test_accuracy", "C2"),
        ("refresh_n4000_a1e-3_t10", "After exact\ninverse refresh", "refresh_test_accuracy", "C3"),
        ("exact_n4000_a1e-3_t10", "Exact\nalpha=1e-3", "test_accuracy", "C0"),
        ("exact_n4000_a1e-2_t10", "Exact\nalpha=1e-2", "test_accuracy", "C4"),
        ("exact_center_zero_n4000_a1e-3_t10", "Exact\nzero bias", "test_accuracy", "C5"),
    ]
    vals, ses, labels, colors = [], [], [], []
    for cell, label, metric, color in probe_cells:
        group = diag_grouped[cell]
        accs = [100 * _metric(r, metric) for r in group]
        vals.append(mean(accs))
        ses.append(_se(accs))
        labels.append(label)
        colors.append(color)
    xs = list(range(len(labels)))
    ax.bar(xs, vals, yerr=ses, capsize=3, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 65)
    ax.set_ylabel("accuracy at 4000 neurons (%)")
    ax.set_title("Mechanism probes (2 seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    for x, val in zip(xs, vals):
        ax.text(x, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 5 -- Streaming continual boundary")
    out = FIG_DIR / "cl_f5_streaming_continual.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ main
FIGURES = {
    "F1": figure_f1,
    "F2": figure_f2,
    "F3": figure_f3,
    "F4": figure_f4,
    "F5": figure_f5,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        action="append",
        default=None,
        help="Generate only the named figures (e.g. --only F2 --only F4)",
    )
    args = p.parse_args()
    todo = args.only or list(FIGURES)
    for name in todo:
        if name not in FIGURES:
            print(f"unknown figure: {name}")
            continue
        try:
            out = FIGURES[name]()
            print(f"  {name}: {out.relative_to(ROOT)}")
        except Exception as e:
            print(f"  {name}: FAILED -- {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
