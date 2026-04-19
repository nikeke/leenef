"""Benchmark: MC return alternatives and EMA recentering.

Compares five target modes on LunarLander-v3:
  1. MC (baseline)
  2. MC + EMA recentering
  3. n-step returns (n=20, n=50)
  4. Eligibility traces (streaming MC equivalent)
  5. Differential (eligibility + EMA reward baseline)

All configs: 4000 neurons, β=0.999, solve every 10, 1000 episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from leenef.rl import NEFFQIAgent  # noqa: E402


def run_experiment(
    config: dict,
    n_episodes: int = 1000,
    warmup_episodes: int = 20,
    seed: int = 0,
) -> dict:
    """Run a single experiment and return results."""
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)

    agent = NEFFQIAgent(
        d_obs=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        seed=seed,
        **config,
    )

    # Warmup
    agent.warmup(env=env, n_episodes=warmup_episodes)

    rewards = []
    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            total_reward += reward
            obs = next_obs
        agent.end_episode()
        rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            recent = np.mean(rewards[-50:])
            elapsed = time.time() - t0
            print(f"  ep {ep + 1:4d}  recent_50={recent:7.1f}  time={elapsed:.0f}s")

    elapsed = time.time() - t0
    env.close()

    rewards_arr = np.array(rewards)
    window = min(50, len(rewards_arr))
    best_w = max(
        np.mean(rewards_arr[i : i + window]) for i in range(len(rewards_arr) - window + 1)
    )
    final_w = np.mean(rewards_arr[-window:])

    return {
        "best_50": round(float(best_w), 1),
        "final_50": round(float(final_w), 1),
        "time_s": round(elapsed, 1),
        "recentered": getattr(agent, "_recentered_total", 0),
        "rewards": [round(float(r), 1) for r in rewards],
    }


def main():
    parser = argparse.ArgumentParser(description="Target mode benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--neurons", type=int, default=4000)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["mc", "mc_ema", "nstep20", "nstep50", "eligibility", "differential"],
    )
    args = parser.parse_args()

    # Base config shared by all experiments
    base = dict(
        n_neurons=args.neurons,
        gamma=0.99,
        alpha=1e-2,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
        solve_every=10,
        forget_factor=0.999,
        recenter_interval=100,
        recenter_percentile=5.0,
    )

    configs = {
        "mc": {
            **base,
            "target_mode": "mc",
        },
        "mc_ema": {
            **base,
            "target_mode": "mc",
            "activity_decay": 0.99,
        },
        "nstep20": {
            **base,
            "target_mode": "nstep",
            "n_step": 20,
        },
        "nstep50": {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
        },
        "eligibility": {
            **base,
            "target_mode": "eligibility",
        },
        "differential": {
            **base,
            "target_mode": "differential",
            "reward_ema_decay": 0.99,
        },
    }

    results = {}
    for name in args.configs:
        if name not in configs:
            print(f"Unknown config: {name}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")
        result = run_experiment(
            configs[name],
            n_episodes=args.episodes,
            seed=args.seed,
        )
        results[name] = result
        print(
            f"  -> best_50={result['best_50']}, final_50={result['final_50']}, "
            f"time={result['time_s']}s, recentered={result['recentered']}"
        )

    # Save results
    os.makedirs("results/rl", exist_ok=True)
    out_path = "results/rl/target_modes.json"
    # Strip per-episode rewards for compact output
    summary = {}
    for name, r in results.items():
        summary[name] = {k: v for k, v in r.items() if k != "rewards"}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'Config':<16} {'best_50':>8} {'final_50':>9} {'time_s':>7} {'recentered':>11}")
    print("-" * 55)
    for name, r in results.items():
        print(
            f"{name:<16} {r['best_50']:>8.1f} {r['final_50']:>9.1f} "
            f"{r['time_s']:>7.0f} {r['recentered']:>11d}"
        )


if __name__ == "__main__":
    main()
