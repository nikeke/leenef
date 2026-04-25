#!/usr/bin/env python3
"""envpool-batched fast path for the NEF-FQI single-knob HP sweep.

Mirrors the CLI and JSON output of ``benchmarks/run_hp_sweep.py`` but
hosts ``N`` independent ``NEFFQIAgent`` instances in a single Python
process and steps their environments via one batched envpool call.

Each agent in a batch:
  * has its own seed (numpy + torch via ``NEFFQIAgent(seed=...)``)
  * runs ``warmup`` on a private gym env (sequentially, before the loop)
  * trains for ``n_episodes`` against its own envpool sub-env
  * is evaluated greedy against gym every ``eval_every`` episodes (same
    semantics as the gym path)

JSON schema is identical to the gym path's output so
``benchmarks/analyze_rl_sweep.py`` and ``benchmarks/plot_rl_sweeps.py``
work without modification.

Usage::

    python benchmarks/run_hp_sweep_envpool.py \\
        --knob n_step --values 10 20 50 100 \\
        --seeds 0 1 2 3 4 5 6 7 \\
        --batch 8

See ``docs/envpool_port_handoff.md`` for the design and parity
requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure this worktree's repo root takes precedence over any shared editable
# install of ``leenef`` whose finder may otherwise resolve ``benchmarks`` to
# a sibling worktree.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import envpool  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.run_hp_sweep import (  # noqa: E402
    INT_KNOBS,
    evaluate_greedy,
    load_base_config,
    parse_value,
    value_tag,
)
from leenef.rl import NEFFQIAgent  # noqa: E402

# Default per-episode truncation in gymnasium for LunarLander-v3.  envpool's
# default is ``INT_MAX``; we clamp to match the gym path.
DEFAULT_MAX_EPISODE_STEPS = 1000


def _build_run_dict(
    *,
    label: str,
    seed: int,
    knob: str | None,
    value,
    rewards: list[float],
    eval_history: list[dict],
    elapsed: float,
    config: dict,
) -> dict:
    rewards_arr = np.array(rewards)
    window = min(50, len(rewards_arr))
    if len(rewards_arr) >= window and window > 0:
        rolling = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
        best_50 = float(rolling.max())
        final_50 = float(rolling[-1])
    elif len(rewards_arr) > 0:
        best_50 = float(rewards_arr.mean())
        final_50 = best_50
    else:
        best_50 = 0.0
        final_50 = 0.0

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
        "config": config,
        "eval_history": eval_history,
        "rewards": [round(float(r), 1) for r in rewards],
    }


def _atomic_write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def run_envpool_batch(
    *,
    env_name: str,
    n_episodes: int,
    labels: list[str],
    seeds: list[int],
    agent_kwargs_list: list[dict],
    eval_every: int = 50,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    pool_seed: int | None = None,
) -> list[dict]:
    """Run one batch of N agents against an envpool of N envs.

    Returns one run_dict per agent, in the order of ``labels``.
    """
    n = len(labels)
    assert len(seeds) == n and len(agent_kwargs_list) == n

    # --- One-time gym probe for spaces ---
    probe = gym.make(env_name)
    d_obs = probe.observation_space.shape[0]
    n_actions = probe.action_space.n
    probe.close()

    # --- Build agents (clean copies of kwargs; pop sweep-only metadata) ---
    knobs: list[str | None] = []
    values: list = []
    clean_kwargs: list[dict] = []
    agents: list[NEFFQIAgent] = []
    for i in range(n):
        kw = dict(agent_kwargs_list[i])
        knobs.append(kw.pop("_knob", None))
        values.append(kw.pop("_value", None))
        clean_kwargs.append(kw)
        agents.append(NEFFQIAgent(d_obs=d_obs, n_actions=n_actions, seed=seeds[i], **kw))

    # --- Sequential per-agent warmup against private gym envs (matches gym path) ---
    for i in range(n):
        warmup_env = gym.make(env_name)
        agents[i].warmup(env=warmup_env, n_episodes=20)
        warmup_env.close()

    # --- Build envpool batch ---
    if pool_seed is None:
        pool_seed = int(seeds[0])
    pool = envpool.make(
        env_name,
        env_type="gymnasium",
        num_envs=n,
        seed=int(pool_seed),
        max_episode_steps=max_episode_steps,
    )

    rewards_per: list[list[float]] = [[] for _ in range(n)]
    eval_history_per: list[list[dict]] = [[] for _ in range(n)]
    ep_counts = [0] * n
    cur_ep_reward = [0.0] * n
    finished = [False] * n
    elapsed_per: list[float | None] = [None] * n

    batch_obs, _ = pool.reset()
    t_start = time.time()

    while not all(finished):
        # Build action vector; inactive agents emit a placeholder action 0.
        actions = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if not finished[i]:
                actions[i] = agents[i].select_action(batch_obs[i])

        next_obs, rewards, term, trunc, _ = pool.step(actions)

        to_reset: list[int] = []
        for i in range(n):
            if finished[i]:
                continue
            done = bool(term[i] or trunc[i])
            agents[i].update(
                batch_obs[i],
                int(actions[i]),
                float(rewards[i]),
                next_obs[i],
                done,
            )
            cur_ep_reward[i] += float(rewards[i])
            if done:
                agents[i].end_episode()
                rewards_per[i].append(cur_ep_reward[i])
                cur_ep_reward[i] = 0.0
                ep_counts[i] += 1

                if ep_counts[i] % eval_every == 0:
                    eval_ret = evaluate_greedy(agents[i], env_name, 20, seed=10000)
                    eval_history_per[i].append({"episode": ep_counts[i], "eval": eval_ret})
                    r50 = (
                        float(np.mean(rewards_per[i][-50:]))
                        if len(rewards_per[i]) >= 50
                        else float(np.mean(rewards_per[i]))
                    )
                    print(
                        f"  [{labels[i]}] ep {ep_counts[i]:4d}  "
                        f"train_50={r50:7.1f}  eval={eval_ret:7.1f}",
                        flush=True,
                    )

                if ep_counts[i] >= n_episodes:
                    finished[i] = True
                    elapsed_per[i] = time.time() - t_start
                else:
                    to_reset.append(i)

        if to_reset:
            # Override envpool's auto-reset state with an explicit reset so
            # the next ``select_action`` sees a true first-of-episode obs.
            reset_obs, _ = pool.reset(env_id=np.array(to_reset, dtype=np.int32))
            for k, i in enumerate(to_reset):
                next_obs[i] = reset_obs[k]

        batch_obs = next_obs

    pool.close()

    return [
        _build_run_dict(
            label=labels[i],
            seed=seeds[i],
            knob=knobs[i],
            value=values[i],
            rewards=rewards_per[i],
            eval_history=eval_history_per[i],
            elapsed=float(elapsed_per[i] or 0.0),
            config=clean_kwargs[i],
        )
        for i in range(n)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="envpool-batched single-knob HP sweep for NEF-FQI"
    )
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--knob", type=str, required=True, help="Name of the agent kwarg to sweep")
    parser.add_argument(
        "--values", nargs="+", required=True, help="Values to try (parsed as int/float/None)"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="envpool batch size (default: number of seeds)",
    )
    parser.add_argument("--base-config", type=str, default="results/rl/base_config.json")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to results/rl/<knob>_lunarlander.json",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="Per-episode truncation (matches gym v3 default of 1000)",
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

    batch_size = args.batch or len(args.seeds)

    # Plan batches: one batch per knob value, partitioned across seeds in
    # chunks of ``batch_size``.  Skipping fully-cached labels collapses to
    # an empty batch which we drop.
    batches: list[tuple[list[str], list[int], list[dict]]] = []
    for v in parsed_values:
        kw = dict(base)
        kw[args.knob] = v
        kw["_knob"] = args.knob
        kw["_value"] = v
        tag = value_tag(v)

        active_pairs = []
        for s in args.seeds:
            label = f"{args.knob}={tag}_s{s}"
            if label in existing and "error" not in existing[label]:
                print(f"  Skipping {label} (already completed)")
                continue
            active_pairs.append((label, s))

        for start in range(0, len(active_pairs), batch_size):
            chunk = active_pairs[start : start + batch_size]
            labels = [p[0] for p in chunk]
            seeds = [p[1] for p in chunk]
            kwargs_list = [dict(kw) for _ in chunk]
            batches.append((labels, seeds, kwargs_list))

    if not batches:
        print("All configs already completed!")
    else:
        print(f"\n{'=' * 70}")
        print(f"  envpool HP Sweep: {args.env} — knob = {args.knob}")
        n_total = sum(len(b[0]) for b in batches)
        print(
            f"  {n_total} runs across {len(batches)} batch(es) "
            f"(batch_size={batch_size}, values={parsed_values})"
        )
        print(f"  base_config from: {args.base_config}")
        print(f"{'=' * 70}\n")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        for bi, (labels, seeds, kwargs_list) in enumerate(batches):
            print(f"  [batch {bi + 1}/{len(batches)}] running {len(labels)} agents: {labels}")
            t_batch = time.time()
            try:
                results = run_envpool_batch(
                    env_name=args.env,
                    n_episodes=args.episodes,
                    labels=labels,
                    seeds=seeds,
                    agent_kwargs_list=kwargs_list,
                    max_episode_steps=args.max_episode_steps,
                )
            except Exception as e:
                print(f"  [batch {bi + 1}] FAILED: {e}", flush=True)
                for label, seed in zip(labels, seeds):
                    existing[label] = {"label": label, "error": str(e), "seed": seed}
                _atomic_write_json(out_path, existing)
                continue
            for r in results:
                existing[r["label"]] = r
                _atomic_write_json(out_path, existing)
                print(f"  → saved {r['label']} to {out_path}", flush=True)
            print(
                f"  [batch {bi + 1}] done in {time.time() - t_batch:.1f}s ({len(labels)} agents)",
                flush=True,
            )

        print(f"\nAll results saved to {out_path}")

    # Summary grouped by value (mirrors the gym path).
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: {args.env} — {args.knob} (envpool path)")
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


# Silence the unused-import warning ruff would otherwise flag for INT_KNOBS
# (re-exported here so callers have a single import source if needed).
_ = INT_KNOBS

if __name__ == "__main__":
    main()
