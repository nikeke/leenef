"""Generate figures for `docs/SSUP-III.md`."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
DENSE = ROOT / "results" / "rl_paper" / "dense"
SPARSE_GPU = ROOT / "results" / "rl_sparse" / "gpu"


def _se(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    return stdev(xs) / math.sqrt(len(xs))


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = list(data.get("runs", []))
    return [r for r in rows if "error" not in r]


def _groups(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row.get("label") or row.get("variant") or "run")].append(row)
    return out


def _metric(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return mean(vals), _se(vals)


def _history(rows: list[dict]) -> tuple[list[int], list[float], list[float]]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        for item in row.get("eval_history", []):
            by_episode[int(item["episode"])].append(float(item["eval"]))
    episodes = sorted(by_episode)
    vals = [mean(by_episode[episode]) for episode in episodes]
    ses = [_se(by_episode[episode]) for episode in episodes]
    return episodes, vals, ses


def _plot_history(ax, rows: list[dict], *, label: str, color: str) -> None:
    episodes, vals, ses = _history(rows)
    lower = [v - s for v, s in zip(vals, ses)]
    upper = [v + s for v, s in zip(vals, ses)]
    ax.plot(episodes, vals, label=label, color=color, linewidth=2)
    ax.fill_between(episodes, lower, upper, color=color, alpha=0.18, linewidth=0)


def figure_f1_dense_targets() -> Path:
    """Dense LunarLander target-construction summary."""

    bootstrap = _groups(_load(DENSE / "bootstrap_lunar.json"))
    dqn = _groups(_load(DENSE / "lunar_dqn.json"))
    final = _groups(_load(DENSE / "lunar_final.json"))
    ensemble = _groups(_load(DENSE / "ensemble_lunar.json"))
    panels = [
        ("TD bootstrap", bootstrap["lunar_td_bootstrap"], "C3"),
        ("MC\nno bootstrap", bootstrap["lunar_mc_no_bootstrap"], "C1"),
        ("n-step\nno bootstrap", bootstrap["lunar_nstep_no_bootstrap"], "C0"),
        ("DQN\ncontext", dqn["lunar_dqn_256x2"], "C7"),
        ("NEF-AVR\nfinal", final["lunar_nef_final"], "C2"),
        ("Thompson\nensemble", ensemble["lunar_ensemble_thompson3"], "C4"),
        ("Voting\nensemble", ensemble["lunar_ensemble_voting3"], "C5"),
    ]

    vals, ses = zip(*[_metric(rows, "robust_eval") for _, rows, _ in panels])
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    xs = list(range(len(panels)))
    ax.bar(xs, vals, yerr=ses, capsize=3, color=[c for _, _, c in panels])
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.45)
    ax.axhline(200, color="black", linewidth=0.7, linestyle="--", alpha=0.45)
    ax.text(len(panels) - 0.55, 203, "200-return reference", fontsize=8, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _, _ in panels], rotation=20, ha="right")
    ax.set_ylabel("robust evaluation return (mean +/- SE)")
    ax.set_title("Dense-control target construction on LunarLander-v3")
    ax.grid(True, axis="y", alpha=0.25)
    for x, value, se in zip(xs, vals, ses):
        ax.text(x, value + se + 7, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    out = FIG_DIR / "rl_f1_dense_targets.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_f2_dense_learning_curves() -> Path:
    """Learning curves for final NEF-AVR and DQN context."""

    final = _groups(_load(DENSE / "lunar_final.json"))["lunar_nef_final"]
    dqn = _groups(_load(DENSE / "lunar_dqn.json"))["lunar_dqn_256x2"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    _plot_history(ax, final, label="NEF-AVR final", color="C2")
    _plot_history(ax, dqn, label="DQN context", color="C7")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.45)
    ax.axhline(200, color="black", linewidth=0.7, linestyle="--", alpha=0.45)
    ax.set_xlabel("training episode")
    ax.set_ylabel("evaluation return (mean +/- SE)")
    ax.set_title("Dense-control learning curves (10 seeds)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    out = FIG_DIR / "rl_f2_dense_learning_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_f3_sparse_results() -> Path:
    """Sparse-control positive and boundary panels."""

    mc_directional = _groups(_load(SPARSE_GPU / "mountaincar_directional_n6000.json"))[
        "nef_sf_hindsight_directional"
    ]
    mc_full_2000 = _groups(_load(SPARSE_GPU / "mountaincar_full_random_direction_n2000.json"))[
        "nef_sf_hindsight"
    ]
    mc_full_6000 = _groups(_load(SPARSE_GPU / "mountaincar_full_random_direction_n6000.json"))[
        "nef_sf_hindsight"
    ]

    acrobot = _groups(_load(SPARSE_GPU / "acrobot_directional_ff9999_n6000.json"))[
        "nef_sf_hindsight_directional"
    ]
    scan_999 = _groups(_load(SPARSE_GPU / "acrobot_directional_ff0p999_scan_n6000.json"))[
        "nef_sf_hindsight_directional"
    ]
    scan_9995 = _groups(_load(SPARSE_GPU / "acrobot_directional_ff0p9995_scan_n6000.json"))[
        "nef_sf_hindsight_directional"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    mc_panels = [
        ("directional\n6000", mc_directional, "C3"),
        ("full successor\n2000", mc_full_2000, "C2"),
        ("full successor\n6000", mc_full_6000, "C0"),
    ]
    ac_panels = [
        ("ff=0.999\n5 seeds", scan_999, "C3"),
        ("ff=0.9995\n5 seeds", scan_9995, "C1"),
        ("ff=0.9999\n10 seeds", acrobot, "C2"),
    ]
    for ax, title, panels in [
        (axes[0], "MountainCar-v0", mc_panels),
        (axes[1], "Acrobot-v1", ac_panels),
    ]:
        xs = list(range(len(panels)))
        vals, ses = zip(*[_metric(rows, "robust_eval") for _, rows, _ in panels])
        ax.bar(xs, vals, yerr=ses, capsize=3, color=[c for _, _, c in panels])
        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _, _ in panels], fontsize=9)
        ax.set_ylabel("robust evaluation return (mean +/- SE)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        for x, value, se in zip(xs, vals, ses):
            ax.text(x, value + se + 5, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Sparse-control target construction: positives and boundaries")
    out = FIG_DIR / "rl_f3_sparse_results.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_f4_sparse_curves_and_cost() -> Path:
    """Sparse learning curves and GPU cost for MountainCar variants."""

    directional = _groups(_load(SPARSE_GPU / "mountaincar_directional_n6000.json"))[
        "nef_sf_hindsight_directional"
    ]
    full_2000 = _groups(_load(SPARSE_GPU / "mountaincar_full_random_direction_n2000.json"))[
        "nef_sf_hindsight"
    ]
    full_6000 = _groups(_load(SPARSE_GPU / "mountaincar_full_random_direction_n6000.json"))[
        "nef_sf_hindsight"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    _plot_history(ax, directional, label="directional 6000", color="C3")
    _plot_history(ax, full_2000, label="full successor 2000", color="C2")
    _plot_history(ax, full_6000, label="full successor 6000", color="C0")
    ax.set_xlabel("training episode")
    ax.set_ylabel("evaluation return (mean +/- SE)")
    ax.set_title("MountainCar-v0 sparse learning curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    panels = [
        ("directional\n6000", directional, "C3"),
        ("full\n2000", full_2000, "C2"),
        ("full\n6000", full_6000, "C0"),
    ]
    xs = list(range(len(panels)))
    vals = [mean([float(r["time_s"]) for r in rows]) for _, rows, _ in panels]
    ses = [_se([float(r["time_s"]) for r in rows]) for _, rows, _ in panels]
    ax.bar(xs, vals, yerr=ses, capsize=3, color=[c for _, _, c in panels])
    ax.set_xticks(xs)
    ax.set_xticklabels([label for label, _, _ in panels])
    ax.set_ylabel("GPU row time (s, mean +/- SE)")
    ax.set_title("Widening cost")
    ax.grid(True, axis="y", alpha=0.25)
    for x, value, se in zip(xs, vals, ses):
        ax.text(x, value + se + 2, f"{value:.1f}s", ha="center", va="bottom", fontsize=8)

    fig.suptitle("MountainCar needs full-successor target structure, not width")
    out = FIG_DIR / "rl_f4_sparse_curves_cost.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for figure in [
        figure_f1_dense_targets,
        figure_f2_dense_learning_curves,
        figure_f3_sparse_results,
        figure_f4_sparse_curves_and_cost,
    ]:
        print(figure())


if __name__ == "__main__":
    main()
