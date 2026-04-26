#!/usr/bin/env python3
"""Analyze a completed single-knob HP sweep and update the live base config.

Clean restart rule for adopting a new base value:
    adopt if  mean(winner).robust_eval - mean(current_base).robust_eval
               > SE_combined = sqrt(SE_winner^2 + SE_current^2)

If the current base value was not among the sweep values, we still
report per-value stats but do NOT change the base (nothing to compare).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.rl_repro import (
    BASE_CONFIG_PATH,
    RESULTS_DIR,
    env_tag,
    hp_run_context,
    is_current_hp_sweep_result,
    result_robust_eval,
)


def aggregate(
    sweep_json: dict,
    knob: str,
    *,
    base_config: dict[str, object],
    expected_run_context: dict[str, object],
) -> dict:
    by_value: dict = {}
    for r in sweep_json.values():
        if not is_current_hp_sweep_result(
            r,
            base_config=base_config,
            knob=knob,
            expected_run_context=expected_run_context,
        ):
            continue
        re = result_robust_eval(r)
        be = r.get("best_eval")
        fe = r.get("final_eval")
        if re is None:
            continue
        tag = _tag(r.get("value"))
        by_value.setdefault(
            tag,
            {"value": r.get("value"), "robust": [], "best": [], "final": []},
        )
        by_value[tag]["robust"].append(re)
        if be is not None:
            by_value[tag]["best"].append(be)
        if fe is not None:
            by_value[tag]["final"].append(fe)
    stats = {}
    for tag, d in by_value.items():
        re = np.array(d["robust"])
        be = np.array(d["best"]) if d["best"] else re
        fe = np.array(d["final"]) if d["final"] else re
        stats[tag] = {
            "value": d["value"],
            "n": len(re),
            "robust_mean": float(re.mean()),
            "robust_se": float(re.std(ddof=1) / np.sqrt(len(re))) if len(re) > 1 else float("nan"),
            "robust_min": float(re.min()),
            "robust_max": float(re.max()),
            "best_mean": float(be.mean()),
            "best_se": float(be.std(ddof=1) / np.sqrt(len(be))) if len(be) > 1 else float("nan"),
            "best_min": float(be.min()),
            "best_max": float(be.max()),
            "final_mean": float(fe.mean()),
            "final_se": float(fe.std(ddof=1) / np.sqrt(len(fe))) if len(fe) > 1 else float("nan"),
        }
    return stats


def _tag(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True)
    ap.add_argument("--env", default="LunarLander-v3")
    ap.add_argument("--sweep-json", default=None)
    ap.add_argument("--base-config", default=str(BASE_CONFIG_PATH))
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument(
        "--min-seeds", type=int, default=4, help="skip values with fewer completed seeds"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Load current base.
    if os.path.exists(args.base_config):
        with open(args.base_config) as f:
            base = json.load(f)
    else:
        print(f"[analyze] No base config at {args.base_config}; cannot decide adoption")
        return

    sweep_path = args.sweep_json or str(RESULTS_DIR / f"{args.knob}_{env_tag(args.env)}.json")
    if not os.path.exists(sweep_path):
        print(f"[analyze] No sweep results at {sweep_path} — skipping")
        return

    with open(sweep_path) as f:
        sweep = json.load(f)

    expected_run_context = hp_run_context(
        args.env,
        args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )
    stats = aggregate(
        sweep,
        args.knob,
        base_config=base,
        expected_run_context=expected_run_context,
    )
    if not stats:
        print(f"[analyze] No valid runs found for knob={args.knob}")
        return

    ordered = sorted(stats.items(), key=lambda kv: -kv[1]["robust_mean"])
    print(f"\n{'=' * 80}")
    print(f"  ANALYSIS: {args.knob}")
    print(f"{'=' * 80}")
    print(
        f"  {'value':>12}  {'n':>3}  {'robust_mean':>12} {'± SE':>7}  "
        f"{'best_mean':>10}  {'final_mean':>11}"
    )
    for tag, s in ordered:
        print(
            f"  {tag:>12}  {s['n']:>3}  {s['robust_mean']:>12.1f} ± {s['robust_se']:5.1f}  "
            f"{s['best_mean']:>10.1f}  {s['final_mean']:>11.1f}"
        )

    current_value = base.get(args.knob)
    current_tag = _tag(current_value)

    winner_tag, winner = ordered[0]
    print(f"\n  winner: {args.knob}={winner_tag}  (robust_mean={winner['robust_mean']:.1f})")
    print(
        f"  current base: {args.knob}={current_tag}  "
        f"(in sweep: {'yes' if current_tag in stats else 'no'})"
    )

    if current_tag not in stats:
        print("  → current base not in sweep values; keeping base unchanged")
        return

    if winner["n"] < args.min_seeds:
        print(f"  → winner has only {winner['n']} seeds (<{args.min_seeds}); keeping base")
        return

    if winner_tag == current_tag:
        print("  → base is already the winner; no change")
        return

    cur = stats[current_tag]
    delta = winner["robust_mean"] - cur["robust_mean"]
    se_comb = float(np.sqrt(winner["robust_se"] ** 2 + cur["robust_se"] ** 2))
    print(f"  Δ = {delta:.1f}   SE_combined = {se_comb:.1f}   ratio = {delta / se_comb:.2f}")

    if delta > se_comb:
        print(f"  ✅ ADOPT: {args.knob}: {current_value} -> {winner['value']}")
        if not args.dry_run:
            base[args.knob] = winner["value"]
            Path(args.base_config).parent.mkdir(parents=True, exist_ok=True)
            tmp = args.base_config + ".tmp"
            with open(tmp, "w") as f:
                json.dump(base, f, indent=2)
            os.replace(tmp, args.base_config)
            print(f"     wrote {args.base_config}")
    else:
        print("  → Δ within 1 SE_combined; keeping base unchanged")


if __name__ == "__main__":
    main()
