#!/usr/bin/env python3
"""Generic single-knob hyperparameter sweep for NEF-FQI on LunarLander.

Reads the current "live" base config from ``results/rl/base_config.json``
and overrides a single knob with each of the provided values, running
8 seeds per value.  Results are saved incrementally so the sweep is
safe to stop, resume, and re-analyze.

Usage:
    python benchmarks/run_hp_sweep.py --knob n_step --values 10 20 50 100 --parallel 6

Base config lives at ``results/rl/base_config.json`` and is updated
between phases by ``benchmarks/analyze_rl_sweep.py`` when a swept knob
beats the current base by more than 1 combined SE of ``best_eval``.

See ``docs/rl_review.md`` §9+ for the full design.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import gymnasium as gym
import numpy as np

from leenef.rl import NEFFQIAgent

# Built-in default base (used if base_config.json is missing).  Must
# match the post-recentering-sweep decision: 4000n, no recentering,
# broadness=0.
DEFAULT_BASE: dict = {
    "n_neurons": 4000,
    "gamma": 0.99,
    "alpha": 1e-2,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 500,
    "solve_every": 10,
    "target_mode": "nstep",
    "n_step": 50,
    "forget_factor": 0.999,
    "fqi_iters": 1,
    "tau": 0.3,
    "broadness": 0.0,
    "recenter_interval": None,
}

# Integer knobs — values parsed as int rather than float.
INT_KNOBS = {
    "n_neurons",
    "n_step",
    "solve_every",
    "fqi_iters",
    "epsilon_decay",
}


def load_base_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            base = json.load(f)
        print(f"Loaded base config from {path}")
    else:
        base = dict(DEFAULT_BASE)
        print(f"No base config at {path}; using built-in defaults")
    return base


def parse_value(knob: str, raw: str):
    if knob in INT_KNOBS:
        return int(float(raw))
    if raw.lower() in ("none", "null"):
        return None
    return float(raw)


def evaluate_greedy(agent, env_name, n_episodes=20, seed=42):
    env = gym.make(env_name)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        done = False
        while not done:
            q = agent.q_values(obs)
            action = int(q.argmax().item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs
        returns.append(total_reward)
    env.close()
    return round(float(np.mean(returns)), 1)


def run_experiment(label, env_name, n_episodes, seed, agent_kwargs, eval_every=50):
    env = gym.make(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    knob = agent_kwargs.pop("_knob", None)
    value = agent_kwargs.pop("_value", None)

    agent = NEFFQIAgent(d_obs=d_obs, n_actions=n_actions, seed=seed, **agent_kwargs)

    warmup_env = gym.make(env_name)
    agent.warmup(env=warmup_env, n_episodes=20)
    warmup_env.close()

    env = gym.make(env_name)
    rewards = []
    eval_history = []
    t_start = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        agent.end_episode()
        rewards.append(total_reward)

        if (ep + 1) % eval_every == 0:
            eval_ret = evaluate_greedy(agent, env_name, 20, seed=10000)
            eval_history.append({"episode": ep + 1, "eval": eval_ret})
            r50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            print(
                f"  [{label}] ep {ep + 1:4d}  train_50={r50:7.1f}  eval={eval_ret:7.1f}",
                flush=True,
            )

    elapsed = time.time() - t_start
    env.close()

    rewards_arr = np.array(rewards)
    window = min(50, len(rewards_arr))
    if len(rewards_arr) >= window:
        rolling = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
        best_50 = float(rolling.max())
        final_50 = float(rolling[-1])
    else:
        best_50 = float(rewards_arr.mean())
        final_50 = best_50

    best_eval = max(e["eval"] for e in eval_history) if eval_history else None
    final_eval = eval_history[-1]["eval"] if eval_history else None

    return {
        "label": label,
        "seed": seed,
        "knob": knob,
        "value": value,
        "best_50": round(best_50, 1),
        "final_50": round(final_50, 1),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "time_s": round(elapsed, 1),
        "config": agent_kwargs,
        "eval_history": eval_history,
        "rewards": [round(float(r), 1) for r in rewards],
    }


def _worker(args):
    label, env_name, n_episodes, seed, agent_kwargs = args
    try:
        return run_experiment(label, env_name, n_episodes, seed, agent_kwargs)
    except Exception as e:
        print(f"  [{label}] FAILED: {e}", flush=True)
        return {"label": label, "error": str(e), "seed": seed}


def value_tag(value) -> str:
    """Filesystem/label-safe stringification of a knob value."""
    if value is None:
        return "none"
    if isinstance(value, float):
        # Keep a few significant digits, strip trailing zeros.
        s = f"{value:g}"
        return s
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Single-knob HP sweep for NEF-FQI")
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--knob", type=str, required=True, help="Name of the agent kwarg to sweep")
    parser.add_argument(
        "--values", nargs="+", required=True, help="Values to try (parsed as int/float/None)"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--base-config", type=str, default="results/rl/base_config.json")
    parser.add_argument(
        "--output", type=str, default=None, help="Defaults to results/rl/<knob>_lunarlander.json"
    )
    args = parser.parse_args()

    base = load_base_config(args.base_config)
    parsed_values = [parse_value(args.knob, v) for v in args.values]

    out_path = args.output or f"results/rl/{args.knob}_lunarlander.json"

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {out_path}")

    tasks = []
    for v in parsed_values:
        kw = dict(base)
        kw[args.knob] = v
        kw["_knob"] = args.knob
        kw["_value"] = v
        tag = value_tag(v)
        for s in args.seeds:
            label = f"{args.knob}={tag}_s{s}"
            if label in existing and "error" not in existing[label]:
                print(f"  Skipping {label} (already completed)")
                continue
            tasks.append((label, args.env, args.episodes, s, kw))

    if not tasks:
        print("All configs already completed!")
    else:
        print(f"\n{'=' * 70}")
        print(f"  HP Sweep: {args.env} — knob = {args.knob}")
        print(f"  {len(tasks)} runs ({len(parsed_values)} values × {len(args.seeds)} seeds)")
        print(f"  values = {parsed_values}")
        print(f"  base_config from: {args.base_config}")
        print(f"  {args.parallel} worker(s)")
        if args.parallel > 1:
            print("  ⚠ Parallel mode: timings NOT comparable to low-load runs")
        print(f"{'=' * 70}\n")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if args.parallel > 1:
            with Pool(args.parallel) as pool:
                for r in pool.imap_unordered(_worker, tasks):
                    if r and "label" in r:
                        existing[r["label"]] = r
                        with open(out_path, "w") as f:
                            json.dump(existing, f, indent=2)
                        print(f"  → saved {r['label']} to {out_path}", flush=True)
        else:
            for t in tasks:
                r = _worker(t)
                if r and "label" in r:
                    existing[r["label"]] = r
                    with open(out_path, "w") as f:
                        json.dump(existing, f, indent=2)
                    print(f"  → saved {r['label']} to {out_path}", flush=True)

        print(f"\nAll results saved to {out_path}")

    # Summary grouped by value (preserves order given on CLI).
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: {args.env} — {args.knob}")
    print(f"{'=' * 80}")
    by_value: dict = {}
    for r in existing.values():
        if "error" in r or "best_eval" not in r or r.get("knob") != args.knob:
            continue
        by_value.setdefault(value_tag(r.get("value")), []).append(r)

    ordered = [value_tag(v) for v in parsed_values if value_tag(v) in by_value]
    ordered += [t for t in sorted(by_value) if t not in ordered]
    for tag in ordered:
        runs = by_value[tag]
        be = [r["best_eval"] for r in runs if r.get("best_eval") is not None]
        fe = [r["final_eval"] for r in runs if r.get("final_eval") is not None]
        if be:
            be_a = np.array(be)
            fe_a = np.array(fe)
            print(
                f"  {args.knob}={tag:>10}  n={len(be)}  "
                f"best={be_a.mean():7.1f} ± {be_a.std(ddof=1) / np.sqrt(len(be)):5.1f} SE  "
                f"final={fe_a.mean():7.1f} ± {fe_a.std(ddof=1) / np.sqrt(len(fe)):5.1f} SE  "
                f"[{min(be):7.1f}, {max(be):7.1f}]"
            )


if __name__ == "__main__":
    main()
