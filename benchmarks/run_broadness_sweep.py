#!/usr/bin/env python3
"""Broadness sweep: test broadness values in [0, 1] with multiple seeds.

Base config: 1×4000n, RLS(β=0.999), n-step(50), recenter/100, p=5%.
Collects per-recenter activity statistics for min_relative analysis.

Usage:
    python benchmarks/run_broadness_sweep.py --parallel 3

    # Specific broadness values only
    python benchmarks/run_broadness_sweep.py --broadness 0.0 0.5 1.0
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import gymnasium as gym
import numpy as np
import torch

from leenef.rl import NEFFQIAgent, NEFFQIEnsemble


def _format_broadness(broadness: float) -> str:
    """Format broadness values without collapsing nearby settings."""
    return f"{broadness:.3f}".rstrip("0").rstrip(".")


def evaluate_greedy(agent, env_name, n_episodes=20, seed=42):
    """Evaluate with greedy policy."""
    env = gym.make(env_name)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        done = False
        while not done:
            if isinstance(agent, NEFFQIEnsemble):
                q_all = torch.stack([m.q_values(obs) for m in agent.members])
                action = int(q_all.mean(dim=0).argmax().item())
            else:
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
    """Run one experiment with eval metrics and recenter logging."""
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
        "broadness": agent_kwargs.get("broadness", 1.0),
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
    """Top-level worker for Pool (must be picklable)."""
    label, env_name, n_episodes, seed, agent_kwargs = args
    try:
        return run_experiment(label, env_name, n_episodes, seed, agent_kwargs)
    except Exception as e:
        print(f"  [{label}] FAILED: {e}", flush=True)
        return {"label": label, "error": str(e), "seed": seed}


def main():
    parser = argparse.ArgumentParser(description="Broadness sweep for NEF-FQI recentering")
    parser.add_argument("--env", type=str, default="LunarLander-v3", help="Gymnasium environment")
    parser.add_argument(
        "--broadness",
        nargs="+",
        type=float,
        default=None,
        help="Specific broadness values (default: [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--neurons", type=int, default=4000)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    broadness_values = args.broadness or [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

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
    }

    out_path = args.output or f"results/rl/broadness_{args.env.split('-')[0].lower()}.json"
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {out_path}")

    tasks = []
    for b in broadness_values:
        for s in args.seeds:
            label = f"b{_format_broadness(b)}_s{s}"
            if label in existing and "error" not in existing[label]:
                print(f"  Skipping {label} (already completed)")
                continue
            kw = dict(base_kwargs)
            kw["broadness"] = b
            tasks.append((label, args.env, args.episodes, s, kw))

    if not tasks:
        print("All configs already completed!")
    else:
        print(f"\n{'=' * 70}")
        print(f"  Broadness Sweep: {args.env}")
        print(f"  {len(tasks)} runs ({len(broadness_values)} values × {len(args.seeds)} seeds)")
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

    # Summary table grouped by broadness
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: {args.env} — Broadness Sweep")
    print(f"{'=' * 90}")
    print(
        f"{'Broadness':>10} {'Seed':>5} {'best_50':>8} {'final_50':>9} "
        f"{'best_eval':>10} {'final_eval':>11} {'time_s':>7} {'recenter':>9}"
    )
    print("-" * 90)

    for b in sorted(set(r.get("broadness", 1.0) for r in existing.values() if "error" not in r)):
        seeds_for_b = sorted(
            [r for r in existing.values() if r.get("broadness") == b and "error" not in r],
            key=lambda r: r.get("seed", 0),
        )
        for r in seeds_for_b:
            b_str = _format_broadness(r["broadness"])
            print(
                f"{b_str:>10} {r.get('seed', '?'):>5} {r['best_50']:>8.1f} {r['final_50']:>9.1f} "
                f"{r.get('best_eval', 'N/A'):>10} {r.get('final_eval', 'N/A'):>11} "
                f"{r['time_s']:>7.0f} {r.get('recentered', 0):>9d}"
            )
        if len(seeds_for_b) > 1:
            evals = [r["best_eval"] for r in seeds_for_b if r.get("best_eval") is not None]
            if evals:
                mean_eval = np.mean(evals)
                std_eval = np.std(evals)
                print(f"{'':>10} {'avg':>5} {'':>8} {'':>9} {mean_eval:>10.1f}±{std_eval:<5.1f}")
        print()


if __name__ == "__main__":
    main()
