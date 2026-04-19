#!/usr/bin/env python3
"""Comprehensive NEF-FQI configuration sweep.

Tests all combinations of key axes on multiple environments:
  - Target: n-step (horizons 30, 50, 100) vs MC baseline
  - RLS: off (buffer) vs on (β=0.999)
  - Recentering: off vs on (/100, 5th percentile)
  - EMA activity: off vs on (decay=0.99), only with recentering
  - Thompson ensemble: off vs on (3 members)

Includes both training metrics (best_50/final_50) and periodic greedy
eval for consistent cross-comparison.

Usage:
    # Full LunarLander sweep with 3 parallel workers
    python benchmarks/run_sweep.py --env LunarLander-v3 --parallel 3

    # Specific configs only
    python benchmarks/run_sweep.py --env LunarLander-v3 \\
        --configs n50_rls n50_rls_recenter n50_rls_recenter_ema

    # CartPole validation (best configs)
    python benchmarks/run_sweep.py --env CartPole-v1

    # Ensemble experiments
    python benchmarks/run_sweep.py --env LunarLander-v3 \\
        --configs thompson_n50 thompson_mc
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

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_greedy(agent, env_name, n_episodes=20, seed=42):
    """Evaluate with greedy policy.  Works for both single agent and ensemble."""
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
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
    env.close()
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    label,
    env_name,
    n_episodes,
    seed,
    agent_kwargs,
    ensemble_members=None,
    eval_every=50,
    eval_episodes=20,
):
    """Run one experiment with both training and eval metrics.

    ``ensemble_members`` can be:
    - None — single agent
    - int — ensemble with that many members (Thompson strategy)
    - (int, str) — ensemble with (n_members, explore_strategy)
    """
    env = gym.make(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    if ensemble_members:
        if isinstance(ensemble_members, tuple):
            n_mem, strategy = ensemble_members
        else:
            n_mem, strategy = ensemble_members, "thompson"
        agent = NEFFQIEnsemble(
            n_members=n_mem,
            explore_strategy=strategy,
            d_obs=d_obs,
            n_actions=n_actions,
            base_seed=seed,
            **agent_kwargs,
        )
    else:
        agent = NEFFQIAgent(
            d_obs=d_obs,
            n_actions=n_actions,
            seed=seed,
            **agent_kwargs,
        )

    # Warmup
    warmup_env = gym.make(env_name)
    agent.warmup(env=warmup_env, n_episodes=20)
    warmup_env.close()

    # Training
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
            eval_ret = evaluate_greedy(agent, env_name, eval_episodes, seed=10000)
            eval_history.append({"episode": ep + 1, "eval": round(eval_ret, 1)})
            r50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            print(
                f"  [{label}] ep {ep + 1:4d}  train_50={r50:7.1f}  eval={eval_ret:7.1f}",
                flush=True,
            )

    elapsed = time.time() - t_start
    env.close()

    # Compute training metrics
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

    recentered = 0
    if ensemble_members:
        recentered = sum(getattr(m, "_recentered_total", 0) for m in agent.members)
        n_mem_out = n_mem
    else:
        recentered = getattr(agent, "_recentered_total", 0)
        n_mem_out = None

    return {
        "label": label,
        "best_50": round(best_50, 1),
        "final_50": round(final_50, 1),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "time_s": round(elapsed, 1),
        "recentered": recentered,
        "config": agent_kwargs,
        "ensemble_members": n_mem_out,
        "eval_history": eval_history,
        "rewards": [round(float(r), 1) for r in rewards],
    }


# ---------------------------------------------------------------------------
# Multiprocessing wrapper
# ---------------------------------------------------------------------------


def _worker(args):
    """Top-level worker for Pool.map (must be picklable)."""
    label, env_name, n_episodes, seed, agent_kwargs, ensemble_members = args
    try:
        return run_experiment(
            label,
            env_name,
            n_episodes,
            seed,
            agent_kwargs,
            ensemble_members=ensemble_members,
        )
    except Exception as e:
        print(f"  [{label}] FAILED: {e}", flush=True)
        return {"label": label, "error": str(e)}


# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

# Base agent kwargs shared across configs (per-env overrides below)
_BASE_LUNAR = dict(
    n_neurons=4000,
    gamma=0.99,
    alpha=1e-2,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=500,
    solve_every=10,
)

_BASE_CARTPOLE = dict(
    n_neurons=2000,
    gamma=0.99,
    alpha=1e-2,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=150,
    solve_every=5,
)

_BASE_ACROBOT = dict(
    n_neurons=4000,
    gamma=0.99,
    alpha=1e-2,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=300,
    solve_every=10,
)

_BASE_MOUNTAINCAR = dict(
    n_neurons=4000,
    gamma=0.99,
    alpha=1e-2,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=300,
    solve_every=10,
)


def _get_base(env_name):
    if "CartPole" in env_name:
        return dict(_BASE_CARTPOLE)
    if "Acrobot" in env_name:
        return dict(_BASE_ACROBOT)
    if "MountainCar" in env_name:
        return dict(_BASE_MOUNTAINCAR)
    return dict(_BASE_LUNAR)


def get_all_configs(env_name):
    """Return dict of {label: (agent_kwargs, ensemble_members)}."""
    base = _get_base(env_name)
    configs = {}

    # --- Single agent: n-step ablation ---

    # n50 + RLS + recenter (our current best single-agent)
    configs["n50_rls_recenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        None,
    )

    # n50 + RLS only (no recenter)
    configs["n50_rls"] = (
        {**base, "target_mode": "nstep", "n_step": 50, "forget_factor": 0.999},
        None,
    )

    # n50 + buffer + recenter (no RLS)
    configs["n50_recenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        None,
    )

    # n50 + buffer only (no RLS, no recenter)
    configs["n50_buffer"] = (
        {**base, "target_mode": "nstep", "n_step": 50},
        None,
    )

    # n50 + RLS + recenter + EMA activity
    configs["n50_rls_recenter_ema"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
            "activity_decay": 0.99,
        },
        None,
    )

    # --- Horizon sweep (all with RLS + recenter) ---

    configs["n30_rls_recenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 30,
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        None,
    )

    configs["n100_rls_recenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 100,
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        None,
    )

    configs["n70_rls_recenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 70,
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        None,
    )

    # --- MC baselines for comparison ---

    configs["mc_rls_recenter"] = (
        {
            **base,
            "target_mode": "mc",
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        None,
    )

    configs["mc_rls"] = (
        {**base, "target_mode": "mc", "forget_factor": 0.999},
        None,
    )

    # --- Thompson ensemble ---

    configs["thompson_n50"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        3,
    )

    configs["thompson_mc"] = (
        {
            **base,
            "target_mode": "mc",
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        3,
    )

    # --- Thompson without recentering (replicates §4.7 conditions) ---

    configs["thompson_n50_norecenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
            "forget_factor": 0.999,
        },
        3,
    )

    # --- Voting ensemble (majority vote, Thompson tiebreak) ---

    configs["voting_n50_recenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        (3, "voting"),
    )

    configs["voting_n50_norecenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
            "forget_factor": 0.999,
        },
        (3, "voting"),
    )

    # --- 4-member voting (total 12k with 3000n each) ---

    configs["voting4_n50_recenter"] = (
        {
            **base,
            "target_mode": "nstep",
            "n_step": 50,
            "forget_factor": 0.999,
            "recenter_interval": 100,
            "recenter_percentile": 5.0,
        },
        (4, "voting"),
    )

    return configs


# ---------------------------------------------------------------------------
# Default episodes per environment
# ---------------------------------------------------------------------------

DEFAULT_EPISODES = {
    "CartPole-v1": 500,
    "Acrobot-v1": 500,
    "MountainCar-v0": 1000,
    "LunarLander-v3": 1000,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Comprehensive NEF-FQI configuration sweep")
    parser.add_argument(
        "--env",
        type=str,
        default="LunarLander-v3",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific config labels to run (default: all)",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--neurons", type=int, default=None, help="Override n_neurons in all configs"
    )
    parser.add_argument(
        "--ensemble-neurons",
        type=int,
        default=None,
        help="Override n_neurons for ensemble members",
    )
    parser.add_argument(
        "--ensemble-members", type=int, default=None, help="Override ensemble member count"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (1 = sequential)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/rl/sweep_<env>.json)",
    )
    args = parser.parse_args()

    env_name = args.env
    n_episodes = args.episodes or DEFAULT_EPISODES.get(env_name, 1000)
    seed = args.seed

    all_configs = get_all_configs(env_name)

    # Override neuron counts if requested
    if args.neurons:
        for label, (kw, ens) in all_configs.items():
            if ens is None:  # single-agent configs only
                kw["n_neurons"] = args.neurons
    if args.ensemble_neurons:
        for label, (kw, ens) in all_configs.items():
            if ens is not None:  # ensemble configs only
                kw["n_neurons"] = args.ensemble_neurons
    if args.ensemble_members:
        new_configs = {}
        for label, (kw, ens) in all_configs.items():
            if ens is not None:
                # Preserve strategy from tuple, just override member count
                if isinstance(ens, tuple):
                    new_configs[label] = (kw, (args.ensemble_members, ens[1]))
                else:
                    new_configs[label] = (kw, args.ensemble_members)
            else:
                new_configs[label] = (kw, ens)
        all_configs = new_configs

    if args.configs:
        configs = {k: v for k, v in all_configs.items() if k in args.configs}
        unknown = set(args.configs) - set(all_configs.keys())
        if unknown:
            print(f"WARNING: unknown configs: {unknown}")
            print(f"Available: {list(all_configs.keys())}")
    else:
        configs = all_configs

    # Load existing results to skip completed configs
    out_path = args.output or f"results/rl/sweep_{env_name.split('-')[0].lower()}.json"
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {out_path}")

    # Filter out completed configs
    todo = {}
    for label, (agent_kwargs, ens) in configs.items():
        if label in existing and "error" not in existing[label]:
            print(f"  Skipping {label} (already completed)")
        else:
            todo[label] = (agent_kwargs, ens)

    if not todo:
        print("All configs already completed!")
    else:
        print(f"\n{'=' * 70}")
        print(f"  NEF-FQI Sweep: {env_name}")
        print(f"  {len(todo)} configs to run, {args.parallel} worker(s)")
        if args.parallel > 1:
            print("  ⚠ Parallel mode: timings NOT comparable to low-load runs")
        print(f"{'=' * 70}\n")

        # Build task list
        tasks = []
        for label, (agent_kwargs, ens) in todo.items():
            tasks.append((label, env_name, n_episodes, seed, agent_kwargs, ens))

        # Run — save after each config finishes (not just at the end)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if args.parallel > 1:
            with Pool(args.parallel) as pool:
                for r in pool.imap_unordered(_worker, tasks):
                    if r and "label" in r:
                        existing[r["label"]] = r
                        with open(out_path, "w") as f:
                            json.dump(existing, f, indent=2)
                        print(f"  → saved {r['label']} to {out_path}")
        else:
            for t in tasks:
                r = _worker(t)
                if r and "label" in r:
                    existing[r["label"]] = r
                    with open(out_path, "w") as f:
                        json.dump(existing, f, indent=2)
                    print(f"  → saved {r['label']} to {out_path}")

        print(f"\nAll results saved to {out_path}")

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY: {env_name}")
    print(f"{'=' * 90}")
    print(
        f"{'Config':<25} {'best_50':>8} {'final_50':>9} "
        f"{'best_eval':>10} {'final_eval':>11} {'time_s':>7} {'recenter':>9}"
    )
    print("-" * 90)
    for label in sorted(existing.keys()):
        r = existing[label]
        if "error" in r:
            print(f"{label:<25} ERROR: {r['error']}")
            continue
        be = r.get("best_eval")
        fe = r.get("final_eval")
        print(
            f"{label:<25} {r['best_50']:>8.1f} {r['final_50']:>9.1f} "
            f"{be if be is not None else 'N/A':>10} "
            f"{fe if fe is not None else 'N/A':>11} "
            f"{r['time_s']:>7.0f} {r.get('recentered', 0):>9d}"
        )


if __name__ == "__main__":
    main()
