#!/usr/bin/env python3
"""Clean NEF-FQI ensemble pilot runner for LunarLander."""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from benchmarks.rl_repro import (
    BASE_CONFIG_PATH,
    EVAL_SEED_BASE,
    RESULTS_DIR,
    can_reuse_result,
    env_tag,
    hp_run_context,
    reset_env,
    result_metadata,
    robust_eval_from_history,
    set_rl_seed,
    training_seed,
    warmup_seed,
)
from benchmarks.run_hp_sweep import _GYMNASIUM_IMPORT_ERROR, gym, load_base_config
from leenef.rl import NEFFQIEnsemble


def ensemble_run_context(
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
        "sweep_type": "ensemble",
        "phase": phase,
    }


def ensemble_config(
    base_config: dict[str, object],
    *,
    n_members: int,
    n_neurons: int,
    train_policy: str,
    eval_policies: list[str],
) -> dict[str, object]:
    config = dict(base_config)
    config["n_neurons"] = n_neurons
    config["n_members"] = n_members
    config["train_policy"] = train_policy
    config["eval_policies"] = list(eval_policies)
    return config


def agent_config(config: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"n_members", "train_policy", "eval_policies"}
    }


def vote_q_values(ensemble: NEFFQIEnsemble, obs: np.ndarray) -> torch.Tensor:
    """Return a one-hot majority-vote score vector for deterministic evaluation."""
    q_all = torch.stack([member.q_values(obs) for member in ensemble.members])
    actions = q_all.argmax(dim=1)
    counts = torch.bincount(actions, minlength=ensemble.members[0].n_actions).float()
    if int((counts == counts.max()).sum().item()) > 1:
        mean_q = q_all.mean(dim=0)
        tied = counts == counts.max()
        counts[tied] += mean_q[tied] - mean_q[tied].min()
    return counts


def policy_q_values(ensemble: NEFFQIEnsemble, obs: np.ndarray, policy: str) -> torch.Tensor:
    if policy == "mean":
        return ensemble.q_values(obs)
    if policy == "voting":
        return vote_q_values(ensemble, obs)
    raise ValueError(f"unknown eval policy: {policy!r}")


def evaluate_policy(
    ensemble: NEFFQIEnsemble,
    env_name: str,
    *,
    policy: str,
    n_episodes: int = 20,
    seed: int = EVAL_SEED_BASE,
) -> float:
    env = gym.make(env_name)
    returns = []
    for i in range(n_episodes):
        obs, _ = reset_env(env, seed=seed + i)
        total_reward = 0.0
        done = False
        while not done:
            q = policy_q_values(ensemble, obs, policy)
            action = int(q.argmax().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    env.close()
    return round(float(np.mean(returns)), 1)


def run_experiment(
    label: str,
    env_name: str,
    n_episodes: int,
    seed: int,
    config: dict[str, object],
    *,
    phase: str,
    eval_every: int = 50,
    eval_episodes: int = 20,
) -> dict[str, object]:
    set_rl_seed(seed)
    env = gym.make(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    n_members = int(config["n_members"])
    train_policy = str(config["train_policy"])
    eval_policies = [str(policy) for policy in config["eval_policies"]]
    member_seed_base = seed * 10_000

    ensemble = NEFFQIEnsemble(
        d_obs=d_obs,
        n_actions=n_actions,
        n_members=n_members,
        explore_strategy=train_policy,
        base_seed=member_seed_base,
        **agent_config(config),
    )

    warmup_seed_value = warmup_seed(seed)
    warmup_env = gym.make(env_name)
    ensemble.warmup(env=warmup_env, n_episodes=20, seed=warmup_seed_value)
    warmup_env.close()

    env = gym.make(env_name)
    rewards = []
    eval_history_by_policy = {policy: [] for policy in eval_policies}
    t_start = time.time()

    for ep in range(n_episodes):
        obs, _ = reset_env(env, seed=training_seed(seed, ep))
        total_reward = 0.0
        done = False
        while not done:
            action = ensemble.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ensemble.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        ensemble.end_episode()
        rewards.append(total_reward)

        if (ep + 1) % eval_every == 0:
            evals = {}
            for policy in eval_policies:
                eval_ret = evaluate_policy(
                    ensemble,
                    env_name,
                    policy=policy,
                    n_episodes=eval_episodes,
                    seed=EVAL_SEED_BASE,
                )
                eval_history_by_policy[policy].append({"episode": ep + 1, "eval": eval_ret})
                evals[policy] = eval_ret
            r50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            eval_text = "  ".join(f"{policy}={value:7.1f}" for policy, value in evals.items())
            print(
                f"  [{label}] ep {ep + 1:4d}  train_50={r50:7.1f}  {eval_text}",
                flush=True,
            )

    elapsed = time.time() - t_start
    env.close()

    rewards_arr = np.array(rewards)
    window = min(50, len(rewards_arr))
    rolling = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
    best_50 = float(rolling.max())
    final_50 = float(rolling[-1])

    metrics_by_policy = {}
    primary_policy = eval_policies[0]
    for policy, history in eval_history_by_policy.items():
        values = [float(item["eval"]) for item in history]
        metrics_by_policy[policy] = {
            "best_eval": max(values) if values else None,
            "final_eval": values[-1] if values else None,
            "robust_eval": robust_eval_from_history(history),
            "eval_history": history,
        }

    primary = metrics_by_policy[primary_policy]
    return {
        **result_metadata(
            config,
            seed,
            run_context=ensemble_run_context(
                env_name,
                n_episodes,
                phase=phase,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
            ),
        ),
        "label": label,
        "phase": phase,
        "n_members": n_members,
        "member_seed_base": member_seed_base,
        "train_policy": train_policy,
        "eval_policies": eval_policies,
        "primary_eval_policy": primary_policy,
        "best_50": round(best_50, 1),
        "final_50": round(final_50, 1),
        "best_eval": primary["best_eval"],
        "final_eval": primary["final_eval"],
        "robust_eval": primary["robust_eval"],
        "metrics_by_policy": metrics_by_policy,
        "time_s": round(elapsed, 1),
        "warmup_seed": warmup_seed_value,
        "evaluation_seed_base": EVAL_SEED_BASE,
        "eval_history": primary["eval_history"],
        "rewards": [round(float(r), 1) for r in rewards],
    }


def _worker(args):
    label, env_name, n_episodes, seed, config, phase, eval_every, eval_episodes = args
    try:
        return run_experiment(
            label,
            env_name,
            n_episodes,
            seed,
            config,
            phase=phase,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
        )
    except Exception as e:
        print(f"  [{label}] FAILED: {e}", flush=True)
        return {"label": label, "error": str(e), "seed": seed}


def _mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


def summarize(existing: dict[str, dict]) -> None:
    print(f"\n{'=' * 80}")
    print("  ENSEMBLE SUMMARY")
    print(f"{'=' * 80}")
    by_policy: dict[str, list[float]] = {}
    for row in existing.values():
        if not isinstance(row, dict) or "error" in row:
            continue
        for policy, metrics in (row.get("metrics_by_policy") or {}).items():
            robust = metrics.get("robust_eval")
            if robust is not None:
                by_policy.setdefault(policy, []).append(float(robust))
    for policy in sorted(by_policy):
        values = by_policy[policy]
        mean, se = _mean_se(values)
        print(
            f"  {policy:>8}  n={len(values)}  robust={mean:7.1f} +/- {se:5.1f} SE  "
            f"[range {min(values):7.1f}, {max(values):7.1f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean NEF-FQI ensemble pilot")
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--phase", type=str, default="p8_ensemble_pilot")
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--neurons", type=int, default=6000, help="Neurons per member")
    parser.add_argument(
        "--train-policy", choices=["thompson", "voting", "ucb"], default="thompson"
    )
    parser.add_argument("--eval-policies", nargs="+", default=["mean", "voting"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--base-config", type=str, default=str(BASE_CONFIG_PATH))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if _GYMNASIUM_IMPORT_ERROR is not None:
        raise SystemExit(str(_GYMNASIUM_IMPORT_ERROR))

    base = load_base_config(args.base_config)
    config = ensemble_config(
        base,
        n_members=args.members,
        n_neurons=args.neurons,
        train_policy=args.train_policy,
        eval_policies=args.eval_policies,
    )
    run_context = ensemble_run_context(
        args.env,
        args.episodes,
        phase=args.phase,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )
    out_path = args.output or str(RESULTS_DIR / f"{args.phase}_{env_tag(args.env)}.json")

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {out_path}")

    tasks = []
    for seed in args.seeds:
        label = f"{args.phase}_m{args.members}_k{args.neurons}_{args.train_policy}_s{seed}"
        if label in existing and can_reuse_result(
            existing[label],
            config,
            expected_seed=seed,
            expected_run_context=run_context,
        ):
            print(f"  Skipping {label} (already completed)")
            continue
        if label in existing:
            print(f"  Re-running {label} (stale result metadata)")
        tasks.append(
            (
                label,
                args.env,
                args.episodes,
                seed,
                config,
                args.phase,
                args.eval_every,
                args.eval_episodes,
            )
        )

    if not tasks:
        print("All ensemble runs already completed!")
    else:
        print(f"\n{'=' * 70}")
        print(f"  Ensemble Sweep: {args.env} - phase = {args.phase}")
        print(f"  {len(tasks)} runs ({args.members} x {args.neurons} neurons/member)")
        print(f"  train_policy = {args.train_policy}")
        print(f"  eval_policies = {args.eval_policies}")
        print(f"  seeds = {args.seeds}")
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

    summarize(existing)


if __name__ == "__main__":
    main()
