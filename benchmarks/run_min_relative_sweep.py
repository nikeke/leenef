#!/usr/bin/env python3
"""Selective recentering sweep: vary ``recenter_min_relative``.

Tests the hypothesis that recentering *genuinely underperforming*
neurons (min_relative > 0) is more reliable than the current pure
rotation default (min_relative = 0), which always recenters the bottom
``recenter_percentile`` regardless of how low their activity actually
is.  Per-run activity stats in ``recenter_log`` suggest that at low
broadness the "bottom 5%" is at 55–65% of mean activity — healthy
neurons being needlessly displaced.

Base config (frozen):
    6000n, n-step(50), RLS(β=0.999), solve_every=10,
    recenter_interval=100, recenter_percentile=5.0, broadness=0.0.

Swept:
    recenter_min_relative ∈ {0.0, 0.4, 0.6, 0.8} plus a "disabled"
    baseline with recenter_interval=None.

See ``docs/rl_review.md`` §5 for the design rationale.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import gymnasium as gym
import numpy as np

from leenef.rl import NEFFQIAgent


def _format_mr(mr) -> str:
    if mr is None:
        return "none"
    s = f"{mr:.3f}".rstrip("0").rstrip(".")
    return s or "0"


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
        "min_relative": agent_kwargs.get("recenter_min_relative", 0.0),
        "recenter_enabled": agent_kwargs.get("recenter_interval") is not None,
        "best_50": round(best_50, 1),
        "final_50": round(final_50, 1),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "time_s": round(elapsed, 1),
        "recentered": agent._recentered_total,
        "config": agent_kwargs,
        "eval_history": eval_history,
        "rewards": [round(float(r), 1) for r in rewards],
        "recenter_log": agent._recenter_log,
    }


def _worker(args):
    label, env_name, n_episodes, seed, agent_kwargs = args
    try:
        return run_experiment(label, env_name, n_episodes, seed, agent_kwargs)
    except Exception as e:
        print(f"  [{label}] FAILED: {e}", flush=True)
        return {"label": label, "error": str(e), "seed": seed}


def main():
    parser = argparse.ArgumentParser(description="min_relative sweep for NEF-FQI recentering")
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument(
        "--min-relative",
        nargs="+",
        type=float,
        default=[0.0, 0.4, 0.6, 0.8],
        help="recenter_min_relative values (plus a disabled baseline added automatically)",
    )
    parser.add_argument(
        "--no-disabled-baseline",
        action="store_true",
        help="Skip the recenter-disabled baseline run",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--neurons", type=int, default=6000)
    parser.add_argument("--broadness", type=float, default=0.0)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output", type=str, default="results/rl/min_relative_lunarlander.json")
    args = parser.parse_args()

    base_kwargs = {
        "n_neurons": args.neurons,
        "gamma": 0.99,
        "alpha": 1e-2,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 500,
        "solve_every": 10,
        "target_mode": "nstep",
        "n_step": 50,
        "forget_factor": 0.999,
        "recenter_interval": 100,
        "recenter_percentile": 5.0,
        "broadness": args.broadness,
    }

    out_path = args.output
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {out_path}")

    tasks = []
    conditions: list[tuple[str, dict]] = []
    if not args.no_disabled_baseline:
        kw = dict(base_kwargs)
        kw["recenter_interval"] = None  # fully disables recentering
        conditions.append(("none", kw))
    for mr in args.min_relative:
        kw = dict(base_kwargs)
        kw["recenter_min_relative"] = mr
        conditions.append((_format_mr(mr), kw))

    for name, kw in conditions:
        for s in args.seeds:
            label = f"mr{name}_s{s}"
            if label in existing and "error" not in existing[label]:
                print(f"  Skipping {label} (already completed)")
                continue
            tasks.append((label, args.env, args.episodes, s, kw))

    if not tasks:
        print("All configs already completed!")
    else:
        print(f"\n{'=' * 70}")
        print(f"  min_relative Sweep: {args.env}")
        print(f"  {len(tasks)} runs ({len(conditions)} conditions × {len(args.seeds)} seeds)")
        print(f"  neurons={args.neurons}  broadness={args.broadness}")
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

    # Summary grouped by condition
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: {args.env} — min_relative Sweep")
    print(f"{'=' * 90}")
    by_cond = {}
    for r in existing.values():
        if "error" in r or "best_eval" not in r:
            continue
        cfg = r.get("config", {})
        if cfg.get("recenter_interval") is None:
            name = "none"
        else:
            name = _format_mr(cfg.get("recenter_min_relative", 0.0))
        by_cond.setdefault(name, []).append(r)

    for name in sorted(by_cond):
        runs = by_cond[name]
        be = [r["best_eval"] for r in runs if r.get("best_eval") is not None]
        fe = [r["final_eval"] for r in runs if r.get("final_eval") is not None]
        if be:
            print(
                f"  min_rel={name:>6}  n={len(be)}  "
                f"best={np.mean(be):7.1f} [{min(be):7.1f}, {max(be):7.1f}]  "
                f"final={np.mean(fe):7.1f} [{min(fe):7.1f}, {max(fe):7.1f}]"
            )


if __name__ == "__main__":
    main()
