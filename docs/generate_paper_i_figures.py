"""Generate figures for `docs/paper_i.md`.

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


def _dominant_identity_rows(path: Path) -> list[dict]:
    """Return clean rows from the most common code identity in a mixed panel."""
    rows = _load_rows(path, current_only=True)
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

    ordered = sorted(families.items(), key=lambda kv: -max(m for _, m, _ in kv[1]))
    keep = [f"trace α={a:g}" for a in (1e-4, 1e-3, 1e-2)] + [
        f"fixed α={a:g}" for a in (1e-2, 1e-4)
    ]
    keep_set = set(keep)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style = {
        "trace α=0.0001": ("o-", "C0", 2.5),
        "trace α=0.001": ("s-", "C2", 1.5),
        "trace α=0.01": ("^-", "C8", 1.0),
        "fixed α=0.01": ("D--", "C3", 1.5),
        "fixed α=0.0001": ("v--", "C1", 1.0),
    }
    for label, points in ordered:
        if label not in keep_set:
            continue
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [100 * p[1] for p in points]
        ses = [100 * p[2] for p in points]
        marker, color, lw = style.get(label, ("x:", "gray", 1.0))
        ax.errorbar(xs, ys, yerr=ses, fmt=marker, color=color, lw=lw, capsize=3, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("n_neurons")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("F3 — α policy × n on streaming-NEF (row sMNIST, 5 seeds)")
    ax.set_ylim(85, 99)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    out = FIG_DIR / "f3_alpha_vs_n.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F3b
def figure_f3b() -> Path:
    """Full 100-cell alpha calibration as heat maps."""
    summary = json.loads((RESULTS / "streaming_alpha_sweep_summary.json").read_text())
    cells = summary["cells"]
    alphas = sorted({c["alpha_nominal"] for c in cells})
    scales = ("fixed", "trace")
    widths = sorted({c["n_neurons"] for c in cells})
    lookup = {
        (c["n_neurons"], c["alpha_scale"], c["alpha_nominal"]): 100 * c["mean"] for c in cells
    }

    fig, axes = plt.subplots(
        1, len(widths), figsize=(9.8, 3.8), sharey=True, constrained_layout=True
    )
    if len(widths) == 1:
        axes = [axes]

    vmin = min(lookup.values())
    vmax = max(lookup.values())
    for ax, n in zip(axes, widths):
        matrix = [[lookup[(n, scale, alpha)] for alpha in alphas] for scale in scales]
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_title(f"n = {n:,}")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:g}" for a in alphas], rotation=30, ha="right")
        ax.set_yticks(range(len(scales)))
        ax.set_yticklabels(scales)
        ax.set_xlabel("base α")
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                color = "black" if value > (vmin + vmax) / 2 else "white"
                ax.text(x, y, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)

    axes[0].set_ylabel("α policy")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("test accuracy (%)")
    fig.suptitle("F3b — Full α calibration heat map (row sMNIST, 5 seeds per cell)")
    out = FIG_DIR / "f3b_alpha_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _factors(r: dict) -> dict:
    """Per-cell factor dict — the canonical place for B1's bias_mode/encoder."""
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
    """B1 abs-hinge geometry: 3 datasets × bias × encoder."""
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
    fig.suptitle("F4 — B1 panel: bias placement × encoder × dataset (10 seeds)")
    out = FIG_DIR / "f4_b1_biases.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F5
def figure_f5() -> Path:
    """B7 hinge-placement falsifier: raw/decentered/centered bias gaps."""

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
    fig.suptitle("F5 — B7 falsifier: hinge placement, not pure DC mass")
    out = FIG_DIR / "f5_b7_hinge_falsifier.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F6
def figure_f6() -> Path:
    """B3 encoder-strategy panel."""
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
    fig.suptitle("F6 — B3 encoder strategy × dataset (10 seeds, n=5000)")
    out = FIG_DIR / "f6_b3_encoders.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F7
def figure_f7() -> Path:
    """Stream-NEF row sMNIST: accuracy vs n for {batch_acc, accumulate, woodbury}."""
    rows = _load_rows(RESULTS / "streaming_smnist_row_n_neurons.json")
    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        cfg = r.get("config", {})
        key = (cfg.get("solve_mode"), cfg.get("n_neurons"))
        cells[key].append(100 * _metric(r))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    modes = ("batch_acc", "accumulate", "woodbury")
    style = {"batch_acc": ("o-", "C0"), "accumulate": ("s--", "C2"), "woodbury": ("^:", "C1")}
    for m in modes:
        ns = sorted({n for (mm, n) in cells if mm == m})
        if not ns:
            continue
        ms = [mean(cells[(m, n)]) for n in ns]
        ses = [_se(cells[(m, n)]) for n in ns]
        marker, color = style[m]
        ax.errorbar(ns, ms, yerr=ses, fmt=marker, color=color, lw=1.5, capsize=3, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("n_neurons")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("F7 — Stream-NEF row sMNIST: solve modes coincide (H3)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    out = FIG_DIR / "f7_streaming_row_solve_modes.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F9
def figure_f9() -> Path:
    """B4 MLP comparator: best-cell MLP vs B1 hypersphere NEF vs B3 RF NEF."""
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
            ("NEF hypersphere (B1)", b1_hypersphere_dd, "C0"),
            ("NEF receptive_field (B3)", b3_rf, "C2"),
            ("MLP tuned (B4)", b4_best, "C3"),
        ]
    ):
        ms, ses = zip(*[fn(ds) for ds in datasets])
        offsets = [x + (i - 1) * width for x in xs]
        ax.bar(offsets, ms, width, yerr=ses, capsize=3, label=label, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("F9 — Single-layer comparator: NEF vs tuned MLP")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    out = FIG_DIR / "f9_b4_comparator.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ F8
def figure_f8() -> Path:
    """Streaming-NEF + ESN + LSTM on row vs permuted sMNIST."""
    # Stream-NEF row (10 seeds): {n_neurons -> acc%}
    nef_row = {2000: 97.20, 4000: 97.97, 8000: 98.45}
    nef_row_se = {2000: 0.04, 4000: 0.03, 8000: 0.02}
    nef_perm = {2000: 72.96, 4000: 75.61}
    nef_perm_se = {2000: 0.07, 4000: 0.05}

    # ESN row + permuted: pick best leak/sr cell per n
    def _esn_best(panel_path, min_rows_per_n: int | None = None):
        rows = _dominant_identity_rows(panel_path)
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

    esn_row = _esn_best(RESULTS / "b6_esn_smnist_row_panel.json")
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
        "LSTM did not converge\nin compute budget\nESN n=2000 tier in flight",
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

    fig.suptitle("F8 — Stream-NEF vs ESN vs LSTM (right panel: ESN rerun incomplete)")
    out = FIG_DIR / "f8_streaming_baselines.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ main
FIGURES = {
    "F3": figure_f3,
    "F3B": figure_f3b,
    "F4": figure_f4,
    "F5": figure_f5,
    "F6": figure_f6,
    "F7": figure_f7,
    "F8": figure_f8,
    "F9": figure_f9,
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
