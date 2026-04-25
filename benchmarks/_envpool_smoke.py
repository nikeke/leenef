#!/usr/bin/env python3
"""Smoke test: one NEFFQIAgent on the envpool path vs. the gym path.

Runs the same agent config (base_config from results/rl/base_config.json)
for ``--episodes`` (default 200) under both paths and reports rolling
reward at the end.  Used for the parity check described in the
envpool-port handoff (Step 2).

This is a benchmark-style script, not a unit test.  Run manually::

    python benchmarks/_envpool_smoke.py --episodes 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure this worktree's repo root takes precedence over any shared editable
# install of ``leenef`` whose finder may otherwise resolve ``benchmarks`` to
# a sibling worktree.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from benchmarks.run_hp_sweep import DEFAULT_BASE  # noqa: E402
from benchmarks.run_hp_sweep import run_experiment as run_gym_experiment  # noqa: E402
from benchmarks.run_hp_sweep_envpool import run_envpool_batch  # noqa: E402


def _load_base(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(DEFAULT_BASE)


def main():
    parser = argparse.ArgumentParser(description="envpool vs gym single-agent parity smoke test")
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-config", type=str, default="results/rl/base_config.json")
    parser.add_argument(
        "--skip-gym",
        action="store_true",
        help="Run only the envpool side (use when comparing against cached gym results)",
    )
    args = parser.parse_args()

    base = _load_base(args.base_config)
    label = f"smoke_s{args.seed}"
    agent_kwargs = dict(base)
    agent_kwargs["_knob"] = "smoke"
    agent_kwargs["_value"] = None

    print(f"=== envpool path: 1 agent, {args.episodes} eps, seed={args.seed} ===")
    t0 = time.time()
    [envpool_run] = run_envpool_batch(
        env_name=args.env,
        n_episodes=args.episodes,
        labels=[label],
        seeds=[args.seed],
        agent_kwargs_list=[agent_kwargs],
    )
    envpool_elapsed = time.time() - t0
    envpool_rewards = np.array(envpool_run["rewards"])

    if args.skip_gym:
        gym_run = None
        gym_elapsed = float("nan")
        gym_rewards = np.array([])
    else:
        print(f"\n=== gym path: 1 agent, {args.episodes} eps, seed={args.seed} ===")
        t0 = time.time()
        gym_run = run_gym_experiment(
            label=label,
            env_name=args.env,
            n_episodes=args.episodes,
            seed=args.seed,
            agent_kwargs=dict(agent_kwargs),
        )
        gym_elapsed = time.time() - t0
        gym_rewards = np.array(gym_run["rewards"])

    def stats(arr):
        if len(arr) == 0:
            return "n/a"
        last50 = arr[-50:] if len(arr) >= 50 else arr
        return (
            f"mean={arr.mean():7.1f}  "
            f"last50={last50.mean():7.1f}  "
            f"max={arr.max():7.1f}  "
            f"min={arr.min():7.1f}"
        )

    print("\n=== summary ===")
    print(f"  envpool ({envpool_elapsed:6.1f}s): {stats(envpool_rewards)}")
    if not args.skip_gym:
        print(f"  gym     ({gym_elapsed:6.1f}s): {stats(gym_rewards)}")
        diff = envpool_rewards[-50:].mean() - gym_rewards[-50:].mean()
        print(f"  Δ last50 mean (envpool - gym) = {diff:+.1f}")
        print(f"  envpool best_eval = {envpool_run.get('best_eval')}")
        print(f"  gym     best_eval = {gym_run.get('best_eval')}")


if __name__ == "__main__":
    main()
