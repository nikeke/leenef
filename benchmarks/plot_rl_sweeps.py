#!/usr/bin/env python3
"""Generate per-knob sweep plots and a summary panel.

Tolerates partial/in-progress JSONs — skips values with <2 seeds for SE
bars but still plots individual dots.  Safe to run repeatedly during a
live sweep; overwrites PNGs atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.rl_repro import (
    FIGURES_DIR,
    RESULTS_DIR,
    env_tag,
    hp_run_context,
    is_current_hp_sweep_result,
)

KNOBS = [
    "n_step",
    "forget_factor",
    "alpha",
    "fqi_iters",
    "solve_every",
    "tau",
    "n_neurons",
    "gamma",
    "epsilon_decay",
]

LOG_X_KNOBS = {"alpha", "epsilon_decay"}  # use log x-axis for these
# forget_factor is best shown as 1-f on log scale (tail mass).


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2**31


def load_sweep(
    path: str,
    knob: str,
    *,
    base_config: dict[str, object],
    expected_run_context: dict[str, object],
) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    by_value: dict = {}
    for r in data.values():
        if not is_current_hp_sweep_result(
            r,
            base_config=base_config,
            knob=knob,
            expected_run_context=expected_run_context,
        ):
            continue
        be = r.get("best_eval")
        if be is None:
            continue
        v = r.get("value")
        key = (
            (float("inf"), "none")
            if v is None
            else (v if isinstance(v, (int, float)) else 0, str(v))
        )
        by_value.setdefault(key, {"value": v, "best": [], "final": []})
        by_value[key]["best"].append(be)
        fe = r.get("final_eval")
        if fe is not None:
            by_value[key]["final"].append(fe)
    return dict(sorted(by_value.items()))


def _x_positions(by_value: dict, knob: str):
    values = [d["value"] for d in by_value.values()]
    numeric = all(isinstance(v, (int, float)) and v is not None for v in values)
    if not numeric:
        return list(range(len(values))), [str(v) for v in values], False
    if knob == "forget_factor":
        # Plot 1-f on log scale so 0.999 and 0.9995 are readable.
        pos = [1.0 - float(v) for v in values]
        labels = [f"{v:g}" for v in values]
        return pos, labels, True  # log scale
    log_scale = knob in LOG_X_KNOBS
    pos = [float(v) for v in values]
    labels = [f"{v:g}" for v in values]
    return pos, labels, log_scale


def plot_knob(
    knob: str,
    *,
    env_name: str,
    base_config: dict[str, object],
    n_episodes: int,
    eval_every: int,
    eval_episodes: int,
    out_dir: str = str(FIGURES_DIR),
) -> dict | None:
    sweep_path = str(RESULTS_DIR / f"{knob}_{env_tag(env_name)}.json")
    by_value = load_sweep(
        sweep_path,
        knob,
        base_config=base_config,
        expected_run_context=hp_run_context(
            env_name,
            n_episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
        ),
    )
    if not by_value:
        return None

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pos, labels, log_scale = _x_positions(by_value, knob)
    fig, (ax_best, ax_final) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for ax, metric in [(ax_best, "best"), (ax_final, "final")]:
        means, mins, maxs, xs_pts, ys_pts, ns = [], [], [], [], [], []
        for x, (_, d) in zip(pos, by_value.items()):
            vals = d[metric]
            if not vals:
                means.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)
                ns.append(0)
                continue
            arr = np.array(vals)
            means.append(arr.mean())
            mins.append(arr.min())
            maxs.append(arr.max())
            ns.append(len(arr))
            jitter = np.random.default_rng(_stable_seed(knob, metric, x)).random(len(arr)) - 0.5
            if log_scale and x > 0:
                jitter *= x * 0.05
            else:
                jitter *= 0.02 * (max(pos) - min(pos) + 1e-9)
            xs_pts.extend([x + j for j in jitter])
            ys_pts.extend(arr.tolist())

        means_a = np.array(means)
        mins_a = np.array(mins)
        maxs_a = np.array(maxs)

        ax.scatter(xs_pts, ys_pts, alpha=0.35, s=24, color="#4a6aa8", zorder=2)
        valid = ~np.isnan(means_a)
        if valid.any():
            lower = means_a[valid] - mins_a[valid]
            upper = maxs_a[valid] - means_a[valid]
            ax.errorbar(
                np.array(pos)[valid],
                means_a[valid],
                yerr=[lower, upper],
                fmt="o-",
                color="#b34e1e",
                capsize=4,
                linewidth=1.5,
                markersize=6,
                zorder=3,
                label="mean (min-max bars)",
            )
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_ylabel(f"{metric}_eval")
        ax.grid(True, alpha=0.3)
        if metric == "best":
            n_str = ", ".join(str(n) for n in ns)
            ax.set_title(f"NEF-FQI {env_name} — knob: {knob}  (n per value: {n_str})")
            ax.legend(loc="lower right", fontsize=8)

    if log_scale:
        ax_final.set_xscale("log")
    x_label = knob if knob != "forget_factor" else "1 − forget_factor (log)"
    ax_final.set_xlabel(x_label)
    ax_final.set_xticks(pos)
    ax_final.set_xticklabels(labels, rotation=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"sweep_{knob}.png")
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, out_path)

    # Return per-value stats for the summary plot.
    summary = {}
    for _, d in by_value.items():
        if d["best"]:
            arr = np.array(d["best"])
            summary[_tag(d["value"])] = {
                "value": d["value"],
                "n": len(arr),
                "best_mean": float(arr.mean()),
                "best_se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            }
    return summary


def _tag(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def plot_summary(
    per_knob_summary: dict,
    *,
    env_name: str,
    out_dir: str = str(FIGURES_DIR),
) -> None:
    """One-line-per-knob summary: range of means with the best starred."""
    knobs_with_data = [k for k in KNOBS if per_knob_summary.get(k)]
    if not knobs_with_data:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(knobs_with_data) + 1.5))
    yticks = []
    yticklabels = []
    for i, knob in enumerate(knobs_with_data):
        stats = per_knob_summary[knob]
        means = [s["best_mean"] for s in stats.values()]
        best_idx = int(np.argmax(means))
        # Draw each value as a small dot; mark the best.
        for j, (tag, s) in enumerate(stats.items()):
            color = "#b34e1e" if j == best_idx else "#4a6aa8"
            ax.plot(s["best_mean"], i, "o", color=color, markersize=8 if j == best_idx else 5)
            ax.errorbar(s["best_mean"], i, xerr=s["best_se"], color=color, capsize=2, alpha=0.6)
            ax.annotate(
                tag,
                (s["best_mean"], i),
                fontsize=7,
                xytext=(3, -2 if j % 2 == 0 else 6),
                textcoords="offset points",
                color=color,
            )
        yticks.append(i)
        yticklabels.append(knob)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("best_eval mean (± SE)")
    ax.set_title(f"NEF-FQI {env_name} HP sweeps — per-knob mean ± SE of best_eval")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "sweep_summary.png")
    tmp = out_path + ".tmp.png"
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", nargs="+", default=None, help="subset of knobs (default: all)")
    ap.add_argument("--env", default="LunarLander-v3")
    ap.add_argument("--base-config", default=str(RESULTS_DIR / "base_config.json"))
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = ap.parse_args()

    knobs = args.knobs or KNOBS
    with open(args.base_config) as f:
        base_config = json.load(f)
    summary = {}
    for knob in knobs:
        print(f"[plot] {knob} ... ", end="", flush=True)
        s = plot_knob(
            knob,
            env_name=args.env,
            base_config=base_config,
            n_episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            out_dir=args.out_dir,
        )
        if s is None:
            print("no data")
        else:
            summary[knob] = s
            print(f"plotted ({len(s)} values)")

    if len(summary) >= 1:
        plot_summary(summary, env_name=args.env, out_dir=args.out_dir)
        print(f"[plot] summary saved to {args.out_dir}/sweep_summary.png")


if __name__ == "__main__":
    main()
