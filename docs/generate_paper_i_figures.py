"""Generate figures for `docs/SSUP-I.md`.

Reads `results/sl_clean/*.json` and writes PNGs to `docs/figures/`.
Idempotent: regenerate any single figure with `--only F3` etc.

The schema for sl_clean rows is:
    row = {
        "value": float,            # primary metric (e.g. test_accuracy)
        "config": {...},           # full config dict
        "seed": int,
        "metrics": {...},          # detailed per-row metrics
        "wall_time_s": float,
        "code_identity": str,
        ...
    }
Most JSON files are flat dicts keyed by `<phase>_<config>_s<seed>`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sl_clean"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


def _se(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return stdev(xs) / math.sqrt(len(xs))


def _load_rows(path: Path, current_only: bool = False) -> list[dict]:
    """Load a flat-dict-of-rows JSON; tolerate list shapes too.

    If ``current_only`` is set, drop rows whose code_identity carries
    a ``-dirty-`` suffix — those are from uncommitted trees and are
    excluded from headline aggregates per AGENTS.md repro rules.
    """
    data = json.loads(path.read_text())
    rows = list(data.values()) if isinstance(data, dict) else list(data)
    if current_only:
        rows = [r for r in rows if "-dirty-" not in r.get("code_identity", "")]
    return rows


def _dominant_identity_rows(path: Path, *, clean_only: bool = True) -> list[dict]:
    """Return clean rows from the most common code identity in a mixed panel."""
    rows = _load_rows(path, current_only=clean_only)
    counts = Counter(r.get("code_identity", "") for r in rows if r.get("code_identity"))
    if not counts:
        return rows
    identity, _ = counts.most_common(1)[0]
    return [r for r in rows if r.get("code_identity") == identity]


def _group(rows: list[dict], key_fn) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[key_fn(r)].append(r)
    return out


# ------------------------------------------------------------------ F3
def figure_f3() -> Path:
    """α-as-function-of-n curve (streaming-NEF row sMNIST)."""
    summary = json.loads((RESULTS / "streaming_alpha_sweep_summary.json").read_text())
    cells = summary["cells"]

    families: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for c in cells:
        scale = c["alpha_scale"]
        nominal = c["alpha_nominal"]
        label = f"{scale} α={nominal:g}"
        families[label].append((c["n_neurons"], c["mean"], c["se"]))

    keep = [f"fixed α={a:g}" for a in (1e-4, 1e-2)] + [
        f"trace α={a:g}" for a in (1e-4, 1e-3, 1e-2)
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style = {
        "trace α=0.0001": ("o-", "C0", 2.5),
        "trace α=0.001": ("s-", "C2", 1.5),
        "trace α=0.01": ("^-", "C8", 1.0),
        "fixed α=0.01": ("D--", "C3", 1.5),
        "fixed α=0.0001": ("v--", "C1", 1.0),
    }
    for label in keep:
        if label not in families:
            continue
        points = families[label]
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [100 * p[1] for p in points]
        ses = [100 * p[2] for p in points]
        marker, color, lw = style.get(label, ("x:", "gray", 1.0))
        ax.errorbar(xs, ys, yerr=ses, fmt=marker, color=color, lw=lw, capsize=3, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("n_neurons")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Figure 1 — α policy × n on streaming-NEF (row sMNIST, 5 seeds)")
    ax.set_ylim(85, 99)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    out = FIG_DIR / "f1_alpha_vs_n.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _factors(r: dict) -> dict:
    """Per-cell factor dict for bias-mode and encoder analyses."""
    return r.get("metrics", {}).get("factors", {}) or r.get("config", {})


def _metric(r: dict) -> float:
    """Primary metric value, robust to panel rows where `value` holds a label."""
    pm = r.get("primary_metric")
    if pm and pm in r.get("metrics", {}):
        return float(r["metrics"][pm])
    v = r.get("value")
    if isinstance(v, (int, float)):
        return float(v)
    return float(r.get("metrics", {}).get("test_accuracy", 0.0))


# ------------------------------------------------------------------ F4
def figure_f4() -> Path:
    """Abs-hinge geometry: 3 datasets × bias × encoder."""
    datasets = ("mnist", "fashion", "cifar10")
    bias_modes = ("data_driven", "zero", "random_normal")
    encoders = ("hypersphere", "gaussian", "sparse")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, ds in zip(axes, datasets):
        rows = _load_rows(RESULTS / f"biases_{ds}_panel.json")
        cells: dict[tuple, list[float]] = defaultdict(list)
        for r in rows:
            f = _factors(r)
            key = (f.get("bias_mode"), f.get("encoder_strategy"))
            cells[key].append(100 * _metric(r))

        x_pos = list(range(len(encoders)))
        width = 0.27
        for i, bm in enumerate(bias_modes):
            ms, ses = [], []
            for enc in encoders:
                xs = cells.get((bm, enc), [])
                ms.append(mean(xs) if xs else 0.0)
                ses.append(_se(xs))
            offsets = [x + (i - 1) * width for x in x_pos]
            ax.bar(offsets, ms, width, yerr=ses, capsize=3, label=bm)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(encoders, fontsize=9)
        ax.set_title(ds.upper())
        ax.set_ylabel("test accuracy (%)")
        ax.grid(True, axis="y", alpha=0.3)
        if ds == "mnist":
            ax.set_ylim(94, 98)
        elif ds == "fashion":
            ax.set_ylim(85, 90)
        else:
            ax.set_ylim(40, 56)
    axes[0].legend(fontsize=8, loc="lower right", title="bias_mode")
    fig.suptitle("Figure 2 — Bias placement × encoder × dataset (10 seeds)")
    out = FIG_DIR / "f2_biases.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F5
def figure_f5() -> Path:
    """Hinge-placement falsifier: raw/decentered/centered bias gaps."""

    panels = [
        (
            "MNIST",
            [("raw", "biases_mnist_panel"), ("decentered", "biases_mnist_decentered_panel")],
            (94.5, 97.8),
        ),
        (
            "CIFAR-10",
            [("raw", "biases_cifar10_panel"), ("centered", "biases_cifar10_centered_panel")],
            (42.0, 51.0),
        ),
    ]
    modes = ("data_driven", "zero")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    width = 0.34
    for ax, (title, scenarios, ylim) in zip(axes, panels):
        means: dict[tuple[str, str], float] = {}
        ses: dict[tuple[str, str], float] = {}
        for label, stem in scenarios:
            rows = _load_rows(RESULTS / f"{stem}.json")
            cells: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                f = _factors(r)
                if f.get("encoder_strategy") == "hypersphere" and f.get("bias_mode") in modes:
                    cells[f["bias_mode"]].append(100 * _metric(r))
            for mode in modes:
                means[(label, mode)] = mean(cells[mode])
                ses[(label, mode)] = _se(cells[mode])

        xs = list(range(len(scenarios)))
        for i, mode in enumerate(modes):
            offsets = [x + (i - 0.5) * width for x in xs]
            vals = [means[(label, mode)] for label, _ in scenarios]
            errs = [ses[(label, mode)] for label, _ in scenarios]
            ax.bar(
                offsets,
                vals,
                width,
                yerr=errs,
                capsize=3,
                label=mode,
                color=("C0" if i == 0 else "C1"),
            )

        for x, (label, _) in zip(xs, scenarios):
            gap = means[(label, "data_driven")] - means[(label, "zero")]
            y = max(means[(label, mode)] for mode in modes) + 0.12
            ax.text(x, y, f"{gap:+.1f} pp", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _ in scenarios])
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)")
    axes[1].legend(title="bias_mode", fontsize=9, loc="lower right")
    fig.suptitle("Figure 3 — Hinge placement, not pure DC mass")
    out = FIG_DIR / "f3_hinge_falsifier.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F6
def figure_f6() -> Path:
    """Encoder-strategy panel."""
    datasets = ("mnist", "fashion", "cifar10")
    strategies = (
        "hypersphere",
        "gaussian",
        "sparse",
        "receptive_field",
        "whitened",
        "class_contrast",
        "local_pca",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, ds in zip(axes, datasets):
        path = RESULTS / f"b3_strategies_{ds}_panel.json"
        if not path.exists():
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ds.upper())
            continue
        rows = _load_rows(path)
        cells: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            f = _factors(r)
            cells[f.get("encoder_strategy")].append(100 * _metric(r))

        ms = [mean(cells[s]) if cells.get(s) else 0.0 for s in strategies]
        ses = [_se(cells[s]) if cells.get(s) else 0.0 for s in strategies]
        colors = ["C0"] * 3 + ["C2"] + ["C0"] + ["C3"] + ["C1"]
        ax.bar(range(len(strategies)), ms, yerr=ses, capsize=3, color=colors)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
        ax.set_title(ds.upper())
        ax.set_ylabel("test accuracy (%)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Figure 4 — Encoder strategy × dataset (10 seeds, n=5000)")
    out = FIG_DIR / "f4_encoders.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_f_scaling() -> Path:
    """Single-layer width scaling across MNIST/Fashion/CIFAR-10."""
    datasets = [
        ("MNIST", "mnist", (91.5, 98.5)),
        ("Fashion-MNIST", "fashion", (82.0, 90.0)),
        ("CIFAR-10", "cifar10", (42.0, 51.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharex=True)
    for ax, (title, stem, ylim) in zip(axes, datasets):
        rows = _load_rows(RESULTS / f"c1_scaling_{stem}_panel.json")
        cells: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            n = r.get("config", {}).get("n_neurons")
            if n is None:
                raw = str(r.get("value", ""))
                n = int(raw[1:]) if raw.startswith("n") else int(raw)
            cells[int(n)].append(100 * _metric(r))
        ns = sorted(cells)
        ax.errorbar(
            ns,
            [mean(cells[n]) for n in ns],
            yerr=[_se(cells[n]) for n in ns],
            marker="o",
            linewidth=2,
            capsize=3,
            color="C0",
        )
        ax.set_xscale("log")
        ax.set_xlim(400, 25000)
        ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000])
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("n_neurons")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("Figure 5 — Single-layer width scaling (10 seeds)")
    out = FIG_DIR / "f5_scaling.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F9
def figure_f9() -> Path:
    """MLP comparator: best-cell MLP vs hypersphere/RF NEF."""
    datasets = ("mnist", "fashion", "cifar10")

    def b1_hypersphere_dd(ds: str) -> tuple[float, float]:
        rows = _load_rows(RESULTS / f"biases_{ds}_panel.json")
        accs = []
        for r in rows:
            f = _factors(r)
            if f.get("bias_mode") == "data_driven" and f.get("encoder_strategy") == "hypersphere":
                accs.append(100 * _metric(r))
        return mean(accs), _se(accs)

    def b3_rf(ds: str) -> tuple[float, float]:
        rows = _load_rows(RESULTS / f"b3_strategies_{ds}_panel.json")
        accs = []
        for r in rows:
            f = _factors(r)
            if f.get("encoder_strategy") == "receptive_field":
                accs.append(100 * _metric(r))
        return (mean(accs), _se(accs)) if accs else (float("nan"), 0.0)

    def b4_best(ds: str) -> tuple[float, float]:
        rows = _load_rows(RESULTS / f"b4_mlp_{ds}_panel.json")
        per_cell: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            per_cell[r.get("config", {}).get("cell", "?")].append(100 * _metric(r))
        if not per_cell:
            return (float("nan"), 0.0)
        means = {k: mean(v) for k, v in per_cell.items()}
        best_k = max(means, key=means.get)
        return means[best_k], _se(per_cell[best_k])

    width = 0.28
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = list(range(len(datasets)))
    for i, (label, fn, color) in enumerate(
        [
            ("NEF hypersphere", b1_hypersphere_dd, "C0"),
            ("NEF receptive_field", b3_rf, "C2"),
            ("MLP tuned", b4_best, "C3"),
        ]
    ):
        ms, ses = zip(*[fn(ds) for ds in datasets])
        offsets = [x + (i - 1) * width for x in xs]
        ax.bar(offsets, ms, width, yerr=ses, capsize=3, label=label, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Figure 8 — Single-layer comparator: NEF vs tuned MLP")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    out = FIG_DIR / "f8_mlp_comparator.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _static_cell_stats(rows: list[dict], keep_fn) -> tuple[float, float, float]:
    vals = [(100 * _metric(r), float(r.get("wall_time_s", 0.0))) for r in rows if keep_fn(r)]
    accs = [v[0] for v in vals]
    walls = [v[1] for v in vals]
    return mean(accs), _se(accs), mean(walls)


def _b4_best_static(ds: str) -> tuple[str, float, float, float]:
    rows = _load_rows(RESULTS / f"b4_mlp_{ds}_panel.json")
    per_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_cell[r.get("config", {}).get("cell", "?")].append(r)
    means = {
        cell: mean(100 * _metric(r) for r in cell_rows) for cell, cell_rows in per_cell.items()
    }
    best_cell = max(means, key=means.get)
    acc, se, wall = _static_cell_stats(
        rows, lambda r: r.get("config", {}).get("cell") == best_cell
    )
    return best_cell, acc, se, wall


def figure_f_static_wall() -> Path:
    """Figure 7-style wall-clock/accuracy view for the static MLP comparator."""

    dataset_colors = {
        "MNIST": "C0",
        "Fashion-MNIST": "C1",
        "CIFAR-10": "C2",
    }
    marker_by_method = {
        "NEF hypersphere": "o",
        "NEF receptive field": "D",
        "MLP tuned": "^",
    }

    points: list[tuple[str, str, str, float, float, float]] = []
    for title, ds in [("MNIST", "mnist"), ("Fashion-MNIST", "fashion"), ("CIFAR-10", "cifar10")]:
        rows = _load_rows(RESULTS / f"biases_{ds}_panel.json")
        acc, se, wall = _static_cell_stats(
            rows,
            lambda r: (
                _factors(r).get("bias_mode") == "data_driven"
                and _factors(r).get("encoder_strategy") == "hypersphere"
            ),
        )
        label = "Fashion NEF" if title == "Fashion-MNIST" else f"{title} NEF"
        points.append((title, "NEF hypersphere", label, wall, acc, se))

        if ds == "cifar10":
            rf_rows = _load_rows(RESULTS / "b3_strategies_cifar10_panel.json")
            acc, se, wall = _static_cell_stats(
                rf_rows, lambda r: _factors(r).get("encoder_strategy") == "receptive_field"
            )
            points.append((title, "NEF receptive field", "CIFAR-10 RF", wall, acc, se))

        best_cell, acc, se, wall = _b4_best_static(ds)
        label = "Fashion MLP" if title == "Fashion-MNIST" else f"{title} MLP"
        points.append((title, "MLP tuned", label, wall, acc, se))

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for dataset, method, label, wall, acc, se in points:
        ax.errorbar(
            wall,
            acc,
            yerr=se,
            marker=marker_by_method[method],
            color=dataset_colors[dataset],
            capsize=3,
            linestyle="none",
            markersize=7,
        )
        if label == "MNIST MLP":
            xytext = (6, -3)
        elif label == "CIFAR-10 MLP":
            xytext = (6, 3)
        elif label == "CIFAR-10 RF":
            xytext = (6, 8)
        else:
            xytext = (6, 0)
        ax.annotate(label, xy=(wall, acc), xytext=xytext, textcoords="offset points", fontsize=8)

    dataset_handles = [
        Line2D([0], [0], color=color, marker="o", linestyle="none", label=label)
        for label, color in dataset_colors.items()
    ]
    method_handles = [
        Line2D([0], [0], color="0.25", marker=marker, linestyle="none", label=method)
        for method, marker in marker_by_method.items()
    ]
    legend1 = ax.legend(
        handles=dataset_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title="dataset",
        fontsize=8,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.62),
        title="method",
        fontsize=8,
    )

    ax.set_xscale("log")
    ax.set_xlim(3.5, 760)
    ax.set_ylim(47.0, 100.0)
    ax.set_xlabel("wall-clock time per seed (s, CPU)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Figure 9 — Static classifier speed/accuracy trade-off")
    ax.grid(True, which="both", alpha=0.3)
    out = FIG_DIR / "f9_static_wall_accuracy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F8
def figure_f8() -> Path:
    """Streaming-NEF + ESN + LSTM on row vs permuted sMNIST."""
    # Stream-NEF row (10 seeds): {n_neurons -> acc%}
    nef_row = {2000: 97.20, 4000: 97.97, 8000: 98.45}
    nef_row_se = {2000: 0.04, 4000: 0.03, 8000: 0.02}
    nef_perm = {2000: 72.96, 4000: 75.61, 8000: 77.68, 16000: 79.44}
    nef_perm_se = {2000: 0.07, 4000: 0.05, 8000: 0.04, 16000: 0.03}

    # ESN row + permuted: pick best leak/sr cell per n
    def _esn_best(panel_path: Path, min_rows_per_n: int | None = None, *, clean_only: bool = True):
        rows = _dominant_identity_rows(panel_path, clean_only=clean_only)
        cells = defaultdict(list)
        rows_per_n = Counter()
        for r in rows:
            cell = r.get("config", {}).get("cell", "")
            n = r.get("config", {}).get("n_neurons")
            if n is None and cell.startswith("n"):
                try:
                    n = int(cell.split("_")[0][1:])
                except ValueError:
                    continue
            acc = r.get("metrics", {}).get("test_accuracy")
            if acc is not None:
                cells[(n, cell)].append(100.0 * acc)
                rows_per_n[n] += 1
        # best cell per n by mean
        by_n = defaultdict(list)
        for (n, _), accs in cells.items():
            if min_rows_per_n is not None and rows_per_n[n] < min_rows_per_n:
                continue
            by_n[n].append((mean(accs), _se(accs), len(accs)))
        return {n: max(v, key=lambda x: x[0]) for n, v in by_n.items()}

    esn_row = _esn_best(RESULTS / "b6_esn_smnist_row_panel.json", clean_only=False)
    # The current permuted ESN rerun is mixed with an older aborted run.
    # Only plot completed current-identity n-tiers; the n=2000 tier is
    # still in flight and is described in the text.
    esn_perm = _esn_best(RESULTS / "b6_esn_smnist_permuted_panel.json", min_rows_per_n=20)

    # LSTM row best cell (h256_l2_lr0.003 → 99.12 ± 0.09)
    lstm_row = {"acc": 99.12, "se": 0.09}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    # Row panel
    ns_nef_row = sorted(nef_row.keys())
    ax1.errorbar(
        ns_nef_row,
        [nef_row[n] for n in ns_nef_row],
        yerr=[nef_row_se[n] for n in ns_nef_row],
        marker="o",
        linewidth=2,
        label="Stream-NEF",
        color="C0",
    )
    ns_esn_row = sorted(esn_row.keys())
    ax1.errorbar(
        ns_esn_row,
        [esn_row[n][0] for n in ns_esn_row],
        yerr=[esn_row[n][1] for n in ns_esn_row],
        marker="s",
        linewidth=2,
        label="ESN (best cell)",
        color="C1",
    )
    ax1.axhline(
        lstm_row["acc"],
        color="C3",
        linestyle="--",
        linewidth=1.5,
        label=f"LSTM tuned ({lstm_row['acc']:.2f}%)",
    )
    ax1.set_xlabel("n_neurons (reservoir size)")
    ax1.set_ylabel("test accuracy (%)")
    ax1.set_title("Row sMNIST")
    ax1.set_xscale("log")
    ax1.set_ylim(86.0, 99.4)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower right")

    # Permuted panel
    ns_nef_p = sorted(nef_perm.keys())
    ax2.errorbar(
        ns_nef_p,
        [nef_perm[n] for n in ns_nef_p],
        yerr=[nef_perm_se[n] for n in ns_nef_p],
        marker="o",
        linewidth=2,
        label="Stream-NEF",
        color="C0",
    )
    ns_esn_p = sorted(esn_perm.keys())
    ax2.errorbar(
        ns_esn_p,
        [esn_perm[n][0] for n in ns_esn_p],
        yerr=[esn_perm[n][1] for n in ns_esn_p],
        marker="s",
        linewidth=2,
        label="ESN (best cell)",
        color="C1",
    )
    ax2.text(
        0.04,
        0.92,
        "LSTM did not converge\nin the budget on permuted",
        transform=ax2.transAxes,
        fontsize=8,
        color="C3",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    ax2.set_xlabel("n_neurons (reservoir size)")
    ax2.set_ylabel("test accuracy (%)")
    ax2.set_title("Permuted sMNIST")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle("Figure 6 — Streaming sequence accuracy vs width")
    out = FIG_DIR / "f6_streaming_accuracy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_f_wall() -> Path:
    """Wall-clock time vs accuracy on row sMNIST."""
    stream_rows = _load_rows(RESULTS / "streaming_smnist_row_n_neurons.json")
    stream_cells: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in stream_rows:
        n = int(r.get("config", {}).get("n_neurons") or r.get("value"))
        stream_cells[n].append((100 * _metric(r), float(r.get("wall_time_s", 0.0))))

    def esn_best(
        panel_path: Path, *, clean_only: bool = True
    ) -> dict[int, tuple[float, float, float]]:
        rows = _dominant_identity_rows(panel_path, clean_only=clean_only)
        cells: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            cell = r.get("config", {}).get("cell", "")
            n = r.get("config", {}).get("n_neurons")
            if n is None and cell.startswith("n"):
                try:
                    n = int(cell.split("_")[0][1:])
                except ValueError:
                    continue
            acc = r.get("metrics", {}).get("test_accuracy")
            if n is not None and acc is not None:
                cells[(int(n), cell)].append((100 * acc, float(r.get("wall_time_s", 0.0))))
        per_n: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        for (n, _), vals in cells.items():
            accs = [v[0] for v in vals]
            walls = [v[1] for v in vals]
            per_n[n].append((mean(accs), _se(accs), mean(walls)))
        return {n: max(vals, key=lambda x: x[0]) for n, vals in per_n.items()}

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for n, vals in sorted(stream_cells.items()):
        accs = [v[0] for v in vals]
        walls = [v[1] for v in vals]
        wall = mean(walls)
        acc = mean(accs)
        ax.errorbar(wall, acc, yerr=_se(accs), marker="o", color="C0", capsize=3)
        ax.text(wall * 1.05, acc, f"NEF {n // 1000}k", fontsize=8, va="center")

    for n, (acc, se, wall) in sorted(
        esn_best(RESULTS / "b6_esn_smnist_row_panel.json", clean_only=False).items()
    ):
        ax.errorbar(wall, acc, yerr=se, marker="s", color="C1", capsize=3)
        label = f"ESN {n // 1000}k" if n >= 1000 else f"ESN {n}"
        ax.text(wall * 1.05, acc, label, fontsize=8, va="center")

    for label, wall, acc, se in [
        ("LSTM fast", 227, 99.07, 0.03),
        ("LSTM best", 1875, 99.12, 0.09),
    ]:
        ax.errorbar(wall, acc, yerr=se, marker="^", color="C3", capsize=3)
        ax.text(wall * 1.05, acc, label, fontsize=8, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("wall-clock time per seed (s, CPU)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Figure 7 — Row sMNIST speed/accuracy trade-off")
    ax.grid(True, which="both", alpha=0.3)
    out = FIG_DIR / "f7_streaming_wall_accuracy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ main
FIGURES = {
    "F1": figure_f3,
    "F2": figure_f4,
    "F3": figure_f5,
    "F4": figure_f6,
    "F5": figure_f_scaling,
    "F6": figure_f8,
    "F7": figure_f_wall,
    "F8": figure_f9,
    "F9": figure_f_static_wall,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        action="append",
        default=None,
        help="Generate only the named figures (e.g. --only F3 --only F7)",
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
            print(f"  {name}: FAILED — {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
