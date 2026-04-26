#!/usr/bin/env python3
"""Run clean RL sweeps over named full-config overrides."""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from benchmarks.rl_repro import (
    BASE_CONFIG_PATH,
    RESULTS_DIR,
    can_reuse_result,
    env_tag,
    hp_run_context,
)
from benchmarks.run_hp_sweep import load_base_config, run_experiment


def config_grid_run_context(
    env_name: str,
    n_episodes: int,
    *,
    phase: str,
    eval_every: int = 50,
    eval_episodes: int = 20,
) -> dict[str, object]:
    return {
        **hp_run_context(
            env_name,
            n_episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
        ),
        "sweep_type": "config_grid",
        "phase": phase,
    }


def load_config_grid(path: str) -> list[dict[str, object]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("config grid must be a list")
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each config-grid entry must be an object")
        name = item.get("name")
        overrides = item.get("overrides")
        if not isinstance(name, str) or not name:
            raise ValueError("each config-grid entry needs a non-empty string name")
        if name in seen:
            raise ValueError(f"duplicate config-grid name: {name}")
        seen.add(name)
        if not isinstance(overrides, dict):
            raise ValueError(f"config-grid entry {name!r} needs object overrides")
    return data


def merged_config(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    config = dict(base)
    config.update(overrides)
    return config


def _default_output_path(env_name: str, phase: str) -> str:
    return str(RESULTS_DIR / f"{phase}_{env_tag(env_name)}.json")


def _worker(args):
    label, env_name, n_episodes, seed, config_name, config, phase, eval_every = args
    try:
        kwargs = {
            **config,
            "_knob": phase,
            "_value": config_name,
            "_run_context_extra": {
                "sweep_type": "config_grid",
                "phase": phase,
            },
        }
        result = run_experiment(
            label,
            env_name,
            n_episodes,
            seed,
            kwargs,
            eval_every=eval_every,
        )
        result["config_name"] = config_name
        return result
    except Exception as e:
        print(f"  [{label}] FAILED: {e}", flush=True)
        return {"label": label, "error": str(e), "seed": seed}


def summarize(existing: dict[str, dict], config_names: list[str]) -> None:
    print(f"\n{'=' * 80}")
    print("  CONFIG GRID SUMMARY")
    print(f"{'=' * 80}")
    by_name: dict[str, list[dict]] = {}
    for row in existing.values():
        if not isinstance(row, dict) or "error" in row:
            continue
        name = row.get("config_name") or row.get("value")
        if name is not None:
            by_name.setdefault(str(name), []).append(row)

    ordered = [name for name in config_names if name in by_name]
    ordered += [name for name in sorted(by_name) if name not in ordered]
    for name in ordered:
        rows = by_name[name]
        robust = [r["robust_eval"] for r in rows if r.get("robust_eval") is not None]
        best = [r["best_eval"] for r in rows if r.get("best_eval") is not None]
        final = [r["final_eval"] for r in rows if r.get("final_eval") is not None]
        if not robust:
            continue
        robust_arr = np.array(robust)
        best_arr = np.array(best) if best else robust_arr
        final_arr = np.array(final) if final else robust_arr
        robust_se = (
            float(robust_arr.std(ddof=1) / np.sqrt(len(robust_arr)))
            if len(robust_arr) > 1
            else 0.0
        )
        print(
            f"  {name:>16}  n={len(robust):>2}  "
            f"robust={robust_arr.mean():7.1f} +/- {robust_se:5.1f} SE  "
            f"best={best_arr.mean():7.1f}  final={final_arr.mean():7.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Named full-config clean RL sweep")
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--config-grid", type=str, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--base-config", type=str, default=str(BASE_CONFIG_PATH))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    base = load_base_config(args.base_config)
    grid = load_config_grid(args.config_grid)
    out_path = args.output or _default_output_path(args.env, args.phase)
    expected_run_context = config_grid_run_context(
        args.env,
        args.episodes,
        phase=args.phase,
        eval_every=args.eval_every,
    )

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {out_path}")

    tasks = []
    for entry in grid:
        name = str(entry["name"])
        config = merged_config(base, entry["overrides"])
        for seed in args.seeds:
            label = f"{args.phase}_{name}_s{seed}"
            if label in existing and can_reuse_result(
                existing[label],
                config,
                expected_seed=seed,
                expected_run_context=expected_run_context,
            ):
                print(f"  Skipping {label} (already completed)")
                continue
            if label in existing:
                print(f"  Re-running {label} (stale result metadata)")
            tasks.append(
                (label, args.env, args.episodes, seed, name, config, args.phase, args.eval_every)
            )

    if not tasks:
        print("All configs already completed!")
    else:
        print(f"\n{'=' * 70}")
        print(f"  Config Sweep: {args.env} - phase = {args.phase}")
        print(f"  {len(tasks)} runs ({len(grid)} configs x {len(args.seeds)} seeds)")
        print(f"  configs = {[entry['name'] for entry in grid]}")
        print(f"  base_config from: {args.base_config}")
        print(f"  {args.parallel} worker(s)")
        if args.parallel > 1:
            print("  WARNING: parallel mode timings are not comparable to low-load runs")
        print(f"{'=' * 70}\n")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if args.parallel > 1:
            with Pool(args.parallel) as pool:
                for result in pool.imap_unordered(_worker, tasks):
                    if result and "label" in result:
                        existing[result["label"]] = result
                        with open(out_path, "w") as f:
                            json.dump(existing, f, indent=2)
                        print(f"  saved {result['label']} to {out_path}", flush=True)
        else:
            for task in tasks:
                result = _worker(task)
                if result and "label" in result:
                    existing[result["label"]] = result
                    with open(out_path, "w") as f:
                        json.dump(existing, f, indent=2)
                    print(f"  saved {result['label']} to {out_path}", flush=True)

        print(f"\nAll results saved to {out_path}")

    summarize(existing, [str(entry["name"]) for entry in grid])


if __name__ == "__main__":
    main()
