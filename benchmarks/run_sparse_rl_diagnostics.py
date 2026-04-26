#!/usr/bin/env python3
"""Sparse-reward diagnostics for NEF-FQI.

This runner is intentionally small-scale by default.  It records why a sparse
environment is failing before any new algorithmic mechanism is introduced:
whether reward events are discovered, whether targets carry enough diversity,
and whether the normal-equation statistics have enough per-action coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from benchmarks.rl_repro import (
    BASE_CONFIG_PATH,
    EVAL_SEED_BASE,
    RESULTS_DIR,
    env_tag,
    reset_env,
    result_metadata,
    set_rl_seed,
    training_seed,
    warmup_seed,
)
from benchmarks.run_hp_sweep import load_base_config
from leenef.rl import NEFFQIAgent


def _stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=0)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "unique_rounded": int(len({round(float(v), 3) for v in arr})),
        "positive_fraction": round(float((arr > 0).mean()), 6),
        "zero_fraction": round(float((arr == 0).mean()), 6),
    }


def sparse_success_event(
    env_name: str,
    next_obs: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict | None = None,
) -> bool:
    """Return whether this transition reached the sparse task goal."""
    del reward
    tag = env_name.lower()
    info = info or {}
    if "mountaincar" in tag:
        goal_position = float(info.get("goal_position", 0.5))
        return bool(terminated and not truncated and float(next_obs[0]) >= goal_position)
    if "acrobot" in tag:
        return bool(terminated and not truncated)
    if "lunarlander" in tag:
        return bool(info.get("success", False))
    return bool(terminated and not truncated)


def episode_success(env_name: str, total_reward: float, success_events: int) -> bool:
    """Episode-level success metric for sparse/control diagnostics."""
    if success_events > 0:
        return True
    if "lunarlander" in env_name.lower():
        return total_reward >= 200.0
    return False


def compute_targets(
    rewards: list[float],
    *,
    gamma: float,
    target_mode: str,
    n_step: int,
) -> list[float]:
    """Compute model-independent return targets for diagnostics."""
    if target_mode == "nstep":
        out: list[float] = []
        for t in range(len(rewards)):
            horizon = min(n_step, len(rewards) - t)
            g = 0.0
            for k in reversed(range(horizon)):
                g = rewards[t + k] + gamma * g
            out.append(float(g))
        return out

    if target_mode in ("mc", "eligibility"):
        out = []
        g = 0.0
        for reward in reversed(rewards):
            g = reward + gamma * g
            out.append(float(g))
        out.reverse()
        return out

    return [float(r) for r in rewards]


def _distance_to_event_bins(targets: list[float], event_step: int | None) -> dict[str, dict]:
    if event_step is None:
        return {"no_event": _stats(targets)}

    bins: dict[str, list[float]] = {
        "event": [],
        "1_5": [],
        "6_25": [],
        "26_100": [],
        "gt100": [],
        "after_event": [],
    }
    for step, target in enumerate(targets):
        distance = event_step - step
        if distance == 0:
            bins["event"].append(target)
        elif 1 <= distance <= 5:
            bins["1_5"].append(target)
        elif 6 <= distance <= 25:
            bins["6_25"].append(target)
        elif 26 <= distance <= 100:
            bins["26_100"].append(target)
        elif distance > 100:
            bins["gt100"].append(target)
        else:
            bins["after_event"].append(target)
    return {name: _stats(values) for name, values in bins.items() if values}


def diagnose_episode(
    transitions: list[dict[str, object]],
    *,
    env_name: str,
    episode: int,
    gamma: float,
    target_mode: str,
    n_step: int,
    n_actions: int,
) -> dict[str, object]:
    """Summarize one episode for sparse-reward failure analysis."""
    rewards = [float(t["reward"]) for t in transitions]
    actions = [int(t["action"]) for t in transitions]
    success_steps = [i for i, t in enumerate(transitions) if bool(t["success_event"])]
    event_step = success_steps[0] if success_steps else None
    total_reward = float(sum(rewards))
    targets = compute_targets(rewards, gamma=gamma, target_mode=target_mode, n_step=n_step)
    action_counts = np.bincount(actions, minlength=n_actions).astype(int).tolist()

    if not transitions:
        terminal_reason = "empty"
    elif bool(transitions[-1]["truncated"]):
        terminal_reason = "truncated"
    elif success_steps:
        terminal_reason = "success"
    elif bool(transitions[-1]["terminated"]):
        terminal_reason = "terminated"
    else:
        terminal_reason = "unfinished"

    return {
        "episode": episode,
        "reward": round(total_reward, 3),
        "length": len(transitions),
        "success": episode_success(env_name, total_reward, len(success_steps)),
        "success_events": len(success_steps),
        "first_success_step": event_step,
        "positive_rewards": int(sum(r > 0 for r in rewards)),
        "nonzero_rewards": int(sum(r != 0 for r in rewards)),
        "terminal_reason": terminal_reason,
        "action_counts": action_counts,
        "target_stats": _stats(targets),
        "target_by_distance_to_event": _distance_to_event_bins(targets, event_step),
    }


def aggregate_episode_diagnostics(episodes: list[dict[str, object]], n_actions: int) -> dict:
    """Aggregate per-episode sparse diagnostics into run-level metrics."""
    if not episodes:
        return {"episodes": 0}

    rewards = [float(e["reward"]) for e in episodes]
    lengths = [float(e["length"]) for e in episodes]
    successes = [bool(e["success"]) for e in episodes]
    success_indices = [int(e["episode"]) for e in episodes if bool(e["success"])]
    action_counts = np.zeros(n_actions, dtype=np.int64)
    target_unique = []
    for episode in episodes:
        action_counts += np.array(episode["action_counts"], dtype=np.int64)
        target_unique.append(float(episode["target_stats"].get("unique_rounded", 0)))

    return {
        "episodes": len(episodes),
        "reward": _stats(rewards),
        "length": _stats(lengths),
        "success_count": int(sum(successes)),
        "success_rate": round(float(np.mean(successes)), 6),
        "first_success_episode": min(success_indices) if success_indices else None,
        "action_counts": action_counts.astype(int).tolist(),
        "action_fraction": [
            round(float(v), 6) for v in action_counts / max(action_counts.sum(), 1)
        ],
        "target_unique_rounded": _stats(target_unique),
    }


def normal_equation_diagnostics(agent: NEFFQIAgent, max_exact_neurons: int = 512) -> dict:
    """Summarize per-action normal-equation coverage and conditioning."""
    if agent.forget_factor is None:
        return {"mode": "buffer", "buffer_size": len(agent.buffer)}

    actions = []
    for action in range(agent.n_actions):
        ata = agent._ata[action]
        count = int(agent._n_accumulated[action].item())
        trace = float(torch.trace(ata).item())
        reg = agent.alpha * trace / agent.n_neurons
        reg = max(reg, agent.alpha)
        diag = ata.diagonal()
        item: dict[str, object] = {
            "action": action,
            "count": count,
            "trace": round(trace, 6),
            "reg": round(float(reg), 6),
            "diag_min": round(float(diag.min().item()), 6) if diag.numel() else None,
            "diag_max": round(float(diag.max().item()), 6) if diag.numel() else None,
            "condition": None,
        }
        if agent.n_neurons <= max_exact_neurons and count >= 10:
            matrix = ata.clone()
            matrix.diagonal().add_(reg)
            item["condition"] = round(float(torch.linalg.cond(matrix).item()), 6)
        actions.append(item)

    return {
        "mode": "rls",
        "total_count": int(agent._n_accumulated.sum().item()),
        "per_action": actions,
    }


def _evaluate(agent: NEFFQIAgent, env_name: str, n_episodes: int, seed: int) -> dict:
    env = gym.make(env_name)
    rewards = []
    successes = []
    lengths = []
    for i in range(n_episodes):
        obs, _ = reset_env(env, seed=seed + i)
        total_reward = 0.0
        success_events = 0
        done = False
        length = 0
        while not done:
            q = agent.q_values(obs)
            action = int(q.argmax().item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success_events += int(
                sparse_success_event(env_name, next_obs, reward, terminated, truncated, info)
            )
            total_reward += float(reward)
            obs = next_obs
            length += 1
        rewards.append(total_reward)
        successes.append(episode_success(env_name, total_reward, success_events))
        lengths.append(float(length))
    env.close()
    return {
        "reward_mean": round(float(np.mean(rewards)), 3) if rewards else None,
        "success_rate": round(float(np.mean(successes)), 6) if successes else None,
        "length_mean": round(float(np.mean(lengths)), 3) if lengths else None,
    }


def run_diagnostic_experiment(
    label: str,
    env_name: str,
    n_episodes: int,
    seed: int,
    agent_kwargs: dict,
    *,
    eval_every: int = 10,
    eval_episodes: int = 5,
    warmup_episodes: int = 5,
) -> dict:
    """Run a small NEF-FQI diagnostic job and return JSON-safe metrics."""
    set_rl_seed(seed)
    env = gym.make(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    config = dict(agent_kwargs)
    agent = NEFFQIAgent(d_obs=d_obs, n_actions=n_actions, seed=seed, **config)

    if warmup_episodes > 0:
        warmup_env = gym.make(env_name)
        warmup_seed_value = warmup_seed(seed)
        agent.warmup(env=warmup_env, n_episodes=warmup_episodes, seed=warmup_seed_value)
        warmup_env.close()
    else:
        warmup_seed_value = None

    env = gym.make(env_name)
    episode_diagnostics = []
    eval_history = []
    t_start = time.time()

    for ep in range(n_episodes):
        obs, _ = reset_env(env, seed=training_seed(seed, ep))
        transitions = []
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = sparse_success_event(env_name, next_obs, reward, terminated, truncated, info)
            transitions.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success_event": success,
                }
            )
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs

        agent.end_episode()
        episode_diagnostics.append(
            diagnose_episode(
                transitions,
                env_name=env_name,
                episode=ep + 1,
                gamma=float(config.get("gamma", agent.gamma)),
                target_mode=str(config.get("target_mode", agent.target_mode)),
                n_step=int(config.get("n_step", agent.n_step)),
                n_actions=n_actions,
            )
        )

        if eval_every > 0 and (ep + 1) % eval_every == 0:
            eval_result = _evaluate(agent, env_name, eval_episodes, seed=EVAL_SEED_BASE)
            eval_history.append({"episode": ep + 1, **eval_result})
            print(
                f"  [{label}] ep {ep + 1:4d}  "
                f"reward={eval_result['reward_mean']:7.1f}  "
                f"success={eval_result['success_rate']:.2f}",
                flush=True,
            )

    elapsed = time.time() - t_start
    env.close()
    summary = aggregate_episode_diagnostics(episode_diagnostics, n_actions)
    normal_eq = normal_equation_diagnostics(agent)

    return {
        **result_metadata(
            config,
            seed,
            run_context={
                "env_name": env_name,
                "n_episodes": n_episodes,
                "eval_every": eval_every,
                "eval_episodes": eval_episodes,
                "warmup_episodes": warmup_episodes,
                "diagnostic": "sparse_reward",
            },
        ),
        "label": label,
        "env": env_name,
        "time_s": round(elapsed, 1),
        "warmup_seed": warmup_seed_value,
        "evaluation_seed_base": EVAL_SEED_BASE,
        "eval_history": eval_history,
        "episode_diagnostics": episode_diagnostics,
        "diagnostic_summary": summary,
        "normal_equations": normal_eq,
    }


def _default_output_path(env_name: str) -> str:
    return str(RESULTS_DIR / f"sparse_diagnostics_{env_tag(env_name)}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse-reward NEF-FQI diagnostics")
    parser.add_argument("--env", default="MountainCar-v0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument("--base-config", default=str(BASE_CONFIG_PATH))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_base_config(args.base_config)
    label = f"{env_tag(args.env)}_diagnostic_s{args.seed}"
    result = run_diagnostic_experiment(
        label,
        args.env,
        args.episodes,
        args.seed,
        config,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        warmup_episodes=args.warmup_episodes,
    )

    out_path = args.output or _default_output_path(args.env)
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
    existing[label] = result
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved {label} to {out_path}")


if __name__ == "__main__":
    main()
