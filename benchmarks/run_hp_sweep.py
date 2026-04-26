#!/usr/bin/env python3
"""Generic single-knob hyperparameter sweep for NEF-FQI on LunarLander.

Reads the clean restart base config from ``results/rl_clean/base_config.json``
and overrides a single knob with each of the provided values, running
8 seeds per value.  Results are saved incrementally so the sweep is
safe to stop, resume, and re-analyze.

Usage:
    python benchmarks/run_hp_sweep.py --knob n_step --values 10 20 50 100 --parallel 6

Base config lives at ``results/rl_clean/base_config.json`` and is updated
between phases by ``benchmarks/analyze_rl_sweep.py`` when a swept knob
beats the current base by more than 1 combined SE of robust eval
(``mean(eval_history[-3:])``).

See ``docs/rl_clean_sweeps.md`` for the clean rerun design.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from benchmarks.rl_repro import (
    BASE_CONFIG_PATH,
    EVAL_SEED_BASE,
    RESULTS_DIR,
    env_tag,
    hp_expected_config,
    hp_run_context,
    is_current_hp_sweep_result,
    reset_env,
    result_metadata,
    robust_eval_from_history,
    set_rl_seed,
    training_seed,
    warmup_seed,
)
from benchmarks.rl_repro import (
    can_reuse_result as can_reuse_rl_result,
)
from leenef.rl import NEFFQIAgent

_MISSING_GYMNASIUM_MESSAGE = (
    "gymnasium is required to run RL sweeps; install it with "
    "`pip install gymnasium` or install this project's benchmark "
    "dependencies with `pip install .[bench]`."
)
_GYMNASIUM_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    _GYMNASIUM_IMPORT_ERROR = ModuleNotFoundError(_MISSING_GYMNASIUM_MESSAGE)

    class _MissingGymnasium:
        def make(self, *args, **kwargs):
            del args, kwargs
            raise _GYMNASIUM_IMPORT_ERROR

    gym = _MissingGymnasium()

# Built-in default base (used if base_config.json is missing).  Must
# match the post-recentering-sweep decision: 4000n, no recentering,
# broadness=0.
DEFAULT_BASE: dict = {
    "n_neurons": 4000,
    "gamma": 0.97,
    "alpha": 0.1,
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
        obs, _ = reset_env(env, seed=seed + i)
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


def _default_output_path(env_name: str, knob: str) -> str:
    return str(RESULTS_DIR / f"{knob}_{env_tag(env_name)}.json")


def run_experiment(label, env_name, n_episodes, seed, agent_kwargs, eval_every=50):
    set_rl_seed(seed)
    env = gym.make(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    run_kwargs = dict(agent_kwargs)
    knob = run_kwargs.pop("_knob", None)
    value = run_kwargs.pop("_value", None)
    run_context_extra = run_kwargs.pop("_run_context_extra", {})

    agent = NEFFQIAgent(d_obs=d_obs, n_actions=n_actions, seed=seed, **run_kwargs)

    warmup_env = gym.make(env_name)
    warmup_seed_value = warmup_seed(seed)
    agent.warmup(env=warmup_env, n_episodes=20, seed=warmup_seed_value)
    warmup_env.close()

    env = gym.make(env_name)
    rewards = []
    eval_history = []
    t_start = time.time()

    for ep in range(n_episodes):
        obs, _ = reset_env(env, seed=training_seed(seed, ep))
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
            eval_ret = evaluate_greedy(agent, env_name, 20, seed=EVAL_SEED_BASE)
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
    robust_eval = robust_eval_from_history(eval_history)

    return {
        **result_metadata(
            run_kwargs,
            seed,
            run_context={
                **hp_run_context(env_name, n_episodes, eval_every=eval_every),
                **run_context_extra,
            },
        ),
        "label": label,
        "knob": knob,
        "value": value,
        "best_50": round(best_50, 1),
        "final_50": round(final_50, 1),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "robust_eval": robust_eval,
        "time_s": round(elapsed, 1),
        "warmup_seed": warmup_seed_value,
        "evaluation_seed_base": EVAL_SEED_BASE,
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


def can_reuse_result(
    result: dict,
    expected_config: dict,
    *,
    expected_seed: int | None = None,
    expected_run_context: dict[str, object] | None = None,
) -> bool:
    """Return True when an existing result matches the current runner semantics."""
    if expected_seed is None:
        expected_seed = int(result.get("seed", -1))
    return can_reuse_rl_result(
        result,
        expected_config,
        expected_seed=expected_seed,
        expected_run_context=expected_run_context,
    )


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
    parser.add_argument("--base-config", type=str, default=str(BASE_CONFIG_PATH))
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to results/rl_clean/<knob>_<env>.json",
    )
    args = parser.parse_args()
    if _GYMNASIUM_IMPORT_ERROR is not None:
        raise SystemExit(str(_GYMNASIUM_IMPORT_ERROR))

    base = load_base_config(args.base_config)
    parsed_values = [parse_value(args.knob, v) for v in args.values]

    out_path = args.output or _default_output_path(args.env, args.knob)

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
        expected_config = hp_expected_config(base, args.knob, v)
        expected_run_context = hp_run_context(args.env, args.episodes, eval_every=50)
        tag = value_tag(v)
        for s in args.seeds:
            label = f"{args.knob}={tag}_s{s}"
            if label in existing and can_reuse_result(
                existing[label],
                expected_config,
                expected_seed=s,
                expected_run_context=expected_run_context,
            ):
                print(f"  Skipping {label} (already completed)")
                continue
            if label in existing:
                print(f"  Re-running {label} (stale result metadata)")
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
    expected_run_context = hp_run_context(args.env, args.episodes, eval_every=50)
    for r in existing.values():
        if "robust_eval" not in r or not is_current_hp_sweep_result(
            r,
            base_config=base,
            knob=args.knob,
            expected_run_context=expected_run_context,
        ):
            continue
        by_value.setdefault(value_tag(r.get("value")), []).append(r)

    ordered = [value_tag(v) for v in parsed_values if value_tag(v) in by_value]
    ordered += [t for t in sorted(by_value) if t not in ordered]
    for tag in ordered:
        runs = by_value[tag]
        re = [r["robust_eval"] for r in runs if r.get("robust_eval") is not None]
        be = [r["best_eval"] for r in runs if r.get("best_eval") is not None]
        fe = [r["final_eval"] for r in runs if r.get("final_eval") is not None]
        if re:
            re_a = np.array(re)
            be_a = np.array(be)
            fe_a = np.array(fe)
            re_se = float(re_a.std(ddof=1) / np.sqrt(len(re))) if len(re) > 1 else 0.0
            be_se = float(be_a.std(ddof=1) / np.sqrt(len(be))) if len(be) > 1 else 0.0
            fe_se = float(fe_a.std(ddof=1) / np.sqrt(len(fe))) if len(fe) > 1 else 0.0
            print(
                f"  {args.knob}={tag:>10}  n={len(re)}  "
                f"robust={re_a.mean():7.1f} ± {re_se:5.1f} SE  "
                f"best={be_a.mean():7.1f} ± {be_se:5.1f} SE  "
                f"final={fe_a.mean():7.1f} ± {fe_se:5.1f} SE  "
                f"[robust range {min(re):7.1f}, {max(re):7.1f}]"
            )


if __name__ == "__main__":
    main()
