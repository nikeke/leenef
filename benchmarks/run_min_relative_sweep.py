#!/usr/bin/env python3
"""Selective recentering sweep: vary ``recenter_percentile`` × ``recenter_min_relative``.

Tests two orthogonal questions about the dead-neuron heuristic:

1. Is the bottom-5% *cap* (``percentile=5``) useful, or would we rather
   recenter any neuron below an absolute activity floor regardless of
   how many that turns out to be (``percentile=100`` + ``min_relative``)?
2. Does an absolute floor at all matter, i.e. is pure rotation
   (``min_relative=0``) actually suboptimal?

Base config (frozen):
    4000n, n-step(50), RLS(β=0.999), solve_every=10,
    recenter_interval=100, broadness=0.0.

4000n is chosen over 6000n because 4000n wins at 4/5 matched broadness
values in the prior sweep (b=0.0: 142.5 vs 124.9; b=0.2: 152.6 vs 37.0;
b=0.9: 157.4 vs 90.0) and has a much higher worst-case floor.

Broadness is held at 0 here, but it is *not* orthogonal to the
selection rule: in the cumulative-stats branch (default),
``apply_broadness`` scales both ``_act_sum`` and ``_act_count`` by
``(1 - broadness)``, so broadness controls how quickly fresh
activations dominate the stats that ``percentile``/``min_relative``
read.  Revisit once we know which selection rule is best.

Conditions (6 total, 8 seeds each → 48 runs):
    - none            : recenter_interval=None (no recentering)
    - p5_mr0          : current default (pure rotation — bottom 5%)
    - p5_mr0.7        : cap at 5% AND require <70% of mean
    - p100_mr0.4      : no cap, recenter everything below 40% of mean
    - p100_mr0.6      : no cap, recenter everything below 60% of mean
    - p100_mr0.8      : no cap, recenter everything below 80% of mean

Results save incrementally; a 9 h wall-time cap will preserve all
completed runs and resume cleanly on the next invocation.

See ``docs/rl_review.md`` for the design rationale.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import gymnasium as gym
import numpy as np

from leenef.rl import NEFFQIAgent

CONDITIONS: list[tuple[str, dict]] = [
    ("none", {"recenter_interval": None}),
    ("p5_mr0", {"recenter_percentile": 5.0, "recenter_min_relative": 0.0}),
    ("p5_mr0.7", {"recenter_percentile": 5.0, "recenter_min_relative": 0.7}),
    ("p100_mr0.4", {"recenter_percentile": 100.0, "recenter_min_relative": 0.4}),
    ("p100_mr0.6", {"recenter_percentile": 100.0, "recenter_min_relative": 0.6}),
    ("p100_mr0.8", {"recenter_percentile": 100.0, "recenter_min_relative": 0.8}),
]


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

    # Strip internal metadata (e.g. _condition_name) before instantiating the agent.
    condition_name = agent_kwargs.get("_condition_name")
    agent_kwargs_clean = {k: v for k, v in agent_kwargs.items() if not k.startswith("_")}

    agent = NEFFQIAgent(d_obs=d_obs, n_actions=n_actions, seed=seed, **agent_kwargs_clean)

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
        "condition": condition_name,
        "min_relative": agent_kwargs_clean.get("recenter_min_relative", 0.0),
        "percentile": agent_kwargs_clean.get("recenter_percentile", 5.0),
        "recenter_enabled": agent_kwargs_clean.get("recenter_interval") is not None,
        "best_50": round(best_50, 1),
        "final_50": round(final_50, 1),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "time_s": round(elapsed, 1),
        "recentered": agent._recentered_total,
        "config": agent_kwargs_clean,
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
    parser = argparse.ArgumentParser(
        description="percentile × min_relative sweep for NEF-FQI recentering"
    )
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Subset of condition names to run (default: all). "
        f"Available: {[c[0] for c in CONDITIONS]}",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--neurons", type=int, default=4000)
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
        "recenter_min_relative": 0.0,
        "broadness": args.broadness,
    }

    out_path = args.output
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {out_path}")

    # Build condition list
    wanted = args.conditions
    selected = [(n, o) for (n, o) in CONDITIONS if (wanted is None or n in wanted)]
    if wanted:
        unknown = set(wanted) - {n for n, _ in CONDITIONS}
        if unknown:
            parser.error(f"Unknown condition(s): {unknown}")

    tasks = []
    for name, overrides in selected:
        kw = dict(base_kwargs)
        kw.update(overrides)
        kw["_condition_name"] = name
        for s in args.seeds:
            label = f"{name}_s{s}"
            if label in existing and "error" not in existing[label]:
                print(f"  Skipping {label} (already completed)")
                continue
            tasks.append((label, args.env, args.episodes, s, kw))

    if not tasks:
        print("All configs already completed!")
    else:
        print(f"\n{'=' * 70}")
        print(f"  Recentering Sweep: {args.env}")
        print(f"  {len(tasks)} runs ({len(selected)} conditions × {len(args.seeds)} seeds)")
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
    print(f"  SUMMARY: {args.env} — Recentering Sweep")
    print(f"{'=' * 90}")
    by_cond: dict[str, list] = {}
    for r in existing.values():
        if "error" in r or "best_eval" not in r:
            continue
        name = r.get("condition") or "?"
        by_cond.setdefault(name, []).append(r)

    # Preserve the canonical order from CONDITIONS.
    canonical = [n for n, _ in CONDITIONS]
    ordered = [n for n in canonical if n in by_cond] + [
        n for n in sorted(by_cond) if n not in canonical
    ]
    for name in ordered:
        runs = by_cond[name]
        be = [r["best_eval"] for r in runs if r.get("best_eval") is not None]
        fe = [r["final_eval"] for r in runs if r.get("final_eval") is not None]
        nd_last = [
            r["recenter_log"][-1].get("n_dead")
            for r in runs
            if r.get("recenter_log") and isinstance(r["recenter_log"], list) and r["recenter_log"]
        ]
        nd_last = [n for n in nd_last if n is not None]
        n_dead_str = (
            f"  n_dead(last)={np.mean(nd_last):6.0f} [{min(nd_last)}, {max(nd_last)}]"
            if nd_last
            else ""
        )
        if be:
            print(
                f"  {name:>12}  n={len(be)}  "
                f"best={np.mean(be):7.1f} [{min(be):7.1f}, {max(be):7.1f}]  "
                f"final={np.mean(fe):7.1f} [{min(fe):7.1f}, {max(fe):7.1f}]"
                f"{n_dead_str}"
            )


if __name__ == "__main__":
    main()
