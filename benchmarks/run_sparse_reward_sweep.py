#!/usr/bin/env python3
"""Restartable sparse-reward mechanism sweeps for NEF-RL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmarks.rl_repro import (  # noqa: E402
    BASE_CONFIG_PATH,
    EVAL_SEED_BASE,
    code_version,
    env_tag,
    hp_run_context,
    reset_env,
    robust_eval_from_history,
    seed_policy,
    set_rl_seed,
    training_seed,
    warmup_seed,
)
from benchmarks.run_hp_sweep import _GYMNASIUM_IMPORT_ERROR, gym, load_base_config  # noqa: E402
from benchmarks.run_sparse_rl_diagnostics import (  # noqa: E402
    aggregate_episode_diagnostics,
    diagnose_episode,
    sparse_success_event,
)
from leenef.rl import NEFFQIAgent  # noqa: E402
from leenef.rl_successor import NEFSuccessorFeatureAgent  # noqa: E402

RESULT_NAMESPACE = "rl_sparse"
RESULTS_DIR = Path("results") / RESULT_NAMESPACE
RUNNER_VERSION = "rl-sparse-v1"


DEFAULT_CONFIGS: list[dict[str, object]] = [
    {
        "name": "fqi_nstep_env",
        "agent_type": "fqi",
        "reward_signal": "env",
        "overrides": {},
    },
    {
        "name": "fqi_nstep_success",
        "agent_type": "fqi",
        "reward_signal": "success",
        "overrides": {},
    },
    {
        "name": "fqi_event_trace_success",
        "agent_type": "fqi",
        "reward_signal": "success",
        "overrides": {
            "target_mode": "event_trace",
            "event_trace_value": 1.0,
            "event_reward_min": 1.0,
            "event_terminal_only": True,
        },
    },
    {
        "name": "sf_success_k1000",
        "agent_type": "successor",
        "reward_signal": "success",
        "overrides": {
            "n_neurons": 1000,
            "successor_horizon": 100,
            "novelty_bonus": 0.0,
        },
    },
    {
        "name": "sf_novelty_success_k1000",
        "agent_type": "successor",
        "reward_signal": "success",
        "overrides": {
            "n_neurons": 1000,
            "successor_horizon": 100,
            "novelty_bonus": 0.1,
        },
    },
]


def sparse_run_context(
    env_name: str,
    n_episodes: int,
    *,
    phase: str,
    eval_every: int,
    eval_episodes: int,
    warmup_episodes: int,
) -> dict[str, object]:
    return {
        **hp_run_context(
            env_name,
            n_episodes,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
        ),
        "sweep_type": "sparse_reward_mechanisms",
        "phase": phase,
        "warmup_episodes": warmup_episodes,
    }


def sparse_result_metadata(
    config: dict[str, object],
    seed: int,
    *,
    run_context: dict[str, object],
) -> dict[str, object]:
    return {
        "result_namespace": RESULT_NAMESPACE,
        "runner_version": RUNNER_VERSION,
        "config": config,
        "seed": seed,
        "seed_policy": seed_policy(),
        "code_version": code_version(),
        "run_context": run_context,
    }


def can_reuse_sparse_result(
    result: dict,
    expected_config: dict,
    *,
    expected_seed: int,
    expected_run_context: dict[str, object],
) -> bool:
    return (
        "error" not in result
        and result.get("result_namespace") == RESULT_NAMESPACE
        and result.get("runner_version") == RUNNER_VERSION
        and result.get("config") == expected_config
        and result.get("seed") == expected_seed
        and result.get("seed_policy") == seed_policy()
        and result.get("code_version") == code_version()
        and result.get("run_context") == expected_run_context
    )


def build_config(base_config: dict[str, object], entry: dict[str, object]) -> dict[str, object]:
    config = dict(base_config)
    config.update(entry.get("overrides", {}))
    config["agent_type"] = entry["agent_type"]
    config["reward_signal"] = entry["reward_signal"]
    config["config_name"] = entry["name"]
    return config


def _agent_reward(
    env_name: str,
    reward: float,
    next_obs: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict,
    reward_signal: str,
) -> float:
    if reward_signal == "env":
        return float(reward)
    if reward_signal == "success":
        return float(sparse_success_event(env_name, next_obs, reward, terminated, truncated, info))
    raise ValueError(f"unknown reward_signal={reward_signal!r}")


def _fqi_kwargs(config: dict[str, object]) -> dict[str, object]:
    skip = {"agent_type", "reward_signal", "config_name", "successor_horizon", "novelty_bonus"}
    return {key: value for key, value in config.items() if key not in skip}


def _successor_kwargs(config: dict[str, object]) -> dict[str, object]:
    allowed = {
        "n_neurons",
        "gamma",
        "alpha",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay",
        "solve_every",
        "forget_factor",
        "successor_horizon",
        "novelty_bonus",
    }
    kwargs = {key: config[key] for key in allowed if key in config}
    if "successor_horizon" not in kwargs and "n_step" in config:
        kwargs["successor_horizon"] = config["n_step"]
    return kwargs


def make_agent(config: dict[str, object], d_obs: int, n_actions: int, seed: int):
    agent_type = str(config["agent_type"])
    if agent_type == "fqi":
        return NEFFQIAgent(d_obs=d_obs, n_actions=n_actions, seed=seed, **_fqi_kwargs(config))
    if agent_type == "successor":
        return NEFSuccessorFeatureAgent(
            d_obs=d_obs,
            n_actions=n_actions,
            seed=seed,
            **_successor_kwargs(config),
        )
    raise ValueError(f"unknown agent_type={agent_type!r}")


def evaluate_agent(agent, env_name: str, *, n_episodes: int, seed: int) -> dict[str, float]:
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
        successes.append(success_events > 0)
        lengths.append(length)
    env.close()
    return {
        "eval": round(float(np.mean(rewards)), 1),
        "success_rate": round(float(np.mean(successes)), 4),
        "length": round(float(np.mean(lengths)), 1),
    }


def _run_episode(
    agent,
    env,
    env_name: str,
    *,
    seed: int,
    reward_signal: str,
    random_policy: bool,
    rng: np.random.Generator,
) -> tuple[float, dict[str, object]]:
    obs, _ = reset_env(env, seed=seed)
    done = False
    transitions = []
    total_reward = 0.0
    while not done:
        if random_policy:
            action = int(rng.integers(env.action_space.n))
        else:
            action = agent.select_action(obs)
        next_obs, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        success = sparse_success_event(env_name, next_obs, env_reward, terminated, truncated, info)
        train_reward = _agent_reward(
            env_name,
            env_reward,
            next_obs,
            terminated,
            truncated,
            info,
            reward_signal,
        )
        transitions.append(
            {
                "action": action,
                "reward": train_reward,
                "env_reward": float(env_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success_event": success,
            }
        )
        agent.update(obs, action, train_reward, next_obs, done)
        obs = next_obs
        total_reward += float(env_reward)
    return total_reward, transitions


def run_experiment(
    label: str,
    env_name: str,
    n_episodes: int,
    seed: int,
    config: dict[str, object],
    *,
    phase: str,
    eval_every: int,
    eval_episodes: int,
    warmup_episodes: int,
) -> dict[str, object]:
    set_rl_seed(seed)
    env = gym.make(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    agent = make_agent(config, d_obs, n_actions, seed)
    reward_signal = str(config["reward_signal"])
    rng = np.random.default_rng(seed + 17)

    warmup_env = gym.make(env_name)
    warmup_seed_value = warmup_seed(seed)
    for i in range(warmup_episodes):
        _run_episode(
            agent,
            warmup_env,
            env_name,
            seed=warmup_seed_value + i,
            reward_signal=reward_signal,
            random_policy=True,
            rng=rng,
        )
        agent.end_episode()
    warmup_env.close()
    agent.solve()

    env = gym.make(env_name)
    rewards = []
    episode_diagnostics = []
    eval_history = []
    t_start = time.time()
    for ep in range(n_episodes):
        total_reward, transitions = _run_episode(
            agent,
            env,
            env_name,
            seed=training_seed(seed, ep),
            reward_signal=reward_signal,
            random_policy=False,
            rng=rng,
        )
        agent.end_episode()
        rewards.append(total_reward)
        episode_diagnostics.append(
            diagnose_episode(
                transitions,
                env_name=env_name,
                episode=ep + 1,
                gamma=float(config.get("gamma", 0.99)),
                target_mode=str(config.get("target_mode", "nstep")),
                n_step=int(config.get("n_step", config.get("successor_horizon", 1))),
                n_actions=n_actions,
            )
        )

        if (ep + 1) % eval_every == 0:
            eval_result = evaluate_agent(
                agent,
                env_name,
                n_episodes=eval_episodes,
                seed=EVAL_SEED_BASE,
            )
            eval_history.append({"episode": ep + 1, **eval_result})
            recent = np.mean(rewards[-eval_every:])
            print(
                f"  [{label}] ep {ep + 1:4d}  train={recent:7.1f}  "
                f"eval={eval_result['eval']:7.1f}  success={eval_result['success_rate']:.2f}",
                flush=True,
            )

    elapsed = time.time() - t_start
    env.close()
    rewards_arr = np.array(rewards, dtype=float)
    window = min(50, len(rewards_arr))
    rolling = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
    summary = aggregate_episode_diagnostics(episode_diagnostics, n_actions)

    return {
        **sparse_result_metadata(
            config,
            seed,
            run_context=sparse_run_context(
                env_name,
                n_episodes,
                phase=phase,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                warmup_episodes=warmup_episodes,
            ),
        ),
        "label": label,
        "phase": phase,
        "env": env_name,
        "config_name": config["config_name"],
        "agent_type": config["agent_type"],
        "reward_signal": reward_signal,
        "best_50": round(float(rolling.max()), 1),
        "final_50": round(float(rolling[-1]), 1),
        "best_eval": max((e["eval"] for e in eval_history), default=None),
        "final_eval": eval_history[-1]["eval"] if eval_history else None,
        "robust_eval": robust_eval_from_history(eval_history),
        "best_success_rate": max((e["success_rate"] for e in eval_history), default=None),
        "final_success_rate": eval_history[-1]["success_rate"] if eval_history else None,
        "time_s": round(elapsed, 1),
        "warmup_seed": warmup_seed_value,
        "evaluation_seed_base": EVAL_SEED_BASE,
        "eval_history": eval_history,
        "rewards": [round(float(r), 1) for r in rewards],
        "diagnostic_summary": summary,
    }


def _worker(args):
    label, env_name, n_episodes, seed, config, phase, eval_every, eval_episodes, warmup = args
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
            warmup_episodes=warmup,
        )
    except Exception as exc:
        print(f"  [{label}] FAILED: {exc}", flush=True)
        return {"label": label, "error": str(exc), "seed": seed}


def _load_configs(path: str | None) -> list[dict[str, object]]:
    if path is None:
        return DEFAULT_CONFIGS
    with open(path) as f:
        configs = json.load(f)
    if not isinstance(configs, list):
        raise ValueError("config file must contain a list")
    return configs


def _default_output_path(phase: str) -> str:
    return str(RESULTS_DIR / f"{phase}.json")


def summarize(existing: dict[str, dict]) -> None:
    print(f"\n{'=' * 100}")
    print("  SPARSE REWARD SUMMARY")
    print(f"{'=' * 100}")
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in existing.values():
        if not isinstance(row, dict) or "error" in row:
            continue
        groups.setdefault((str(row.get("env")), str(row.get("config_name"))), []).append(row)
    for (env_name, config_name), rows in sorted(groups.items()):
        robust = [r["robust_eval"] for r in rows if r.get("robust_eval") is not None]
        success = [
            r["final_success_rate"] for r in rows if r.get("final_success_rate") is not None
        ]
        if not robust:
            continue
        robust_arr = np.array(robust, dtype=float)
        success_arr = np.array(success, dtype=float) if success else np.array([0.0])
        robust_se = (
            float(robust_arr.std(ddof=1) / np.sqrt(len(robust_arr)))
            if len(robust_arr) > 1
            else 0.0
        )
        success_se = (
            float(success_arr.std(ddof=1) / np.sqrt(len(success_arr)))
            if len(success_arr) > 1
            else 0.0
        )
        print(
            f"  {env_name:<16} {config_name:<28} n={len(rows):>2}  "
            f"robust={robust_arr.mean():8.1f} +/- {robust_se:5.1f}  "
            f"final_success={success_arr.mean():5.2f} +/- {success_se:4.2f}"
        )


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _log_run_progress(completed: int, total: int, started_at: float) -> None:
    elapsed = time.time() - started_at
    rate = elapsed / completed if completed else 0.0
    remaining = rate * (total - completed) if completed else 0.0
    print(
        f"  progress: {completed}/{total} pending runs complete; "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
        flush=True,
    )


def run_sweep(
    *,
    phase: str,
    envs: list[str],
    config_entries: list[dict[str, object]],
    seeds: list[int],
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    warmup_episodes: int,
    parallel: int,
    base_config_path: str,
    output: str | None = None,
) -> dict[str, dict]:
    base = load_base_config(base_config_path)
    out_path = output or _default_output_path(phase)
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {out_path}")

    tasks = []
    for env_name in envs:
        run_context = sparse_run_context(
            env_name,
            episodes,
            phase=phase,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            warmup_episodes=warmup_episodes,
        )
        for entry in config_entries:
            config = build_config(base, entry)
            config_name = str(entry["name"])
            for seed in seeds:
                label = f"{phase}_{env_tag(env_name)}_{config_name}_s{seed}"
                if label in existing and can_reuse_sparse_result(
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
                        env_name,
                        episodes,
                        seed,
                        config,
                        phase,
                        eval_every,
                        eval_episodes,
                        warmup_episodes,
                    )
                )

    print(f"\n{'=' * 80}")
    print(f"  Sparse Reward Sweep: phase={phase}")
    print(f"  envs={envs}")
    print(f"  configs={[entry['name'] for entry in config_entries]}")
    print(f"  seeds={seeds}")
    print(f"  episodes={episodes}  eval_every={eval_every}  eval_episodes={eval_episodes}")
    print(f"  warmup_episodes={warmup_episodes}  workers={parallel}")
    print(f"  output={out_path}")
    print(f"  pending_runs={len(tasks)}")
    print(f"{'=' * 80}\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    completed = 0
    total = len(tasks)
    if tasks:
        if parallel > 1:
            with Pool(parallel) as pool:
                for result in pool.imap_unordered(_worker, tasks):
                    if result and "label" in result:
                        existing[result["label"]] = result
                        with open(out_path, "w") as f:
                            json.dump(existing, f, indent=2)
                        print(f"  saved {result['label']} to {out_path}", flush=True)
                        completed += 1
                        _log_run_progress(completed, total, started_at)
        else:
            for task in tasks:
                result = _worker(task)
                if result and "label" in result:
                    existing[result["label"]] = result
                    with open(out_path, "w") as f:
                        json.dump(existing, f, indent=2)
                    print(f"  saved {result['label']} to {out_path}", flush=True)
                    completed += 1
                    _log_run_progress(completed, total, started_at)

    summarize(existing)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse-reward NEF-RL mechanism sweep")
    parser.add_argument("--phase", default="sparse_p1_mechanisms")
    parser.add_argument("--envs", nargs="+", default=["MountainCar-v0", "Acrobot-v1"])
    parser.add_argument("--configs", default=None, help="Optional JSON config list")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--base-config", default=str(BASE_CONFIG_PATH))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if _GYMNASIUM_IMPORT_ERROR is not None:
        raise SystemExit(str(_GYMNASIUM_IMPORT_ERROR))

    run_sweep(
        phase=args.phase,
        envs=args.envs,
        config_entries=_load_configs(args.configs),
        seeds=args.seeds,
        episodes=args.episodes,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        warmup_episodes=args.warmup_episodes,
        parallel=args.parallel,
        base_config_path=args.base_config,
        output=args.output,
    )


if __name__ == "__main__":
    main()
