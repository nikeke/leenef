#!/usr/bin/env python3
"""envpool-batched fast path for the NEF-FQI single-knob HP sweep.

Mirrors the CLI and JSON output of ``benchmarks/run_hp_sweep.py`` but
hosts ``N`` independent ``NEFFQIAgent`` instances in a single Python
process and steps their environments via one batched envpool call.

Reproducibility model
---------------------

Each label ``s`` is **pinned to envpool sub-env index ``s``** of an
envpool created with a fixed master seed (``--envpool-master-seed``,
default ``0``).  envpool's per-sub-env RNG stream depends only on the
pair ``(master_seed, sub_env_index)`` — not on ``num_envs``, batch
position, or sibling labels — so label ``s``'s training trajectory is
invariant across:

  * different ``--batch`` settings,
  * resuming with different cached labels,
  * different ``--seeds`` lists (as long as ``s`` is in them).

This is *not* trajectory-identical to the gym path (envpool's C++
LunarLander is a different Box2D implementation than gym's, and we
cannot replicate gym's per-episode ``env.reset(seed=seed+ep)``);
parity is checked statistically against ``results/rl/n_step_lunarlander.json``.

JSON schema is identical to the gym path's output so
``benchmarks/analyze_rl_sweep.py`` and ``benchmarks/plot_rl_sweeps.py``
work without modification.

See ``docs/envpool_port_handoff.md`` for the full design and parity
requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Iterator

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

# Default master seed for envpool.  Combined with each label's seed
# ``s`` -> sub-env index ``s``, this gives label-stable trajectories
# across batches and resumes.
DEFAULT_ENVPOOL_MASTER_SEED = 0


def _reorder_by_env_id(
    info: dict | None,
    arrays: tuple,
    expected_ids: np.ndarray,
    *,
    where: str,
) -> tuple:
    """Reorder ``arrays`` so that row ``k`` corresponds to env_id ``expected_ids[k]``.

    Raises
    ------
    RuntimeError
        If ``info`` does not contain ``env_id``, or if the returned
        ``env_id`` is not exactly the set ``expected_ids``.

    Notes
    -----
    envpool's synchronous batch step is *expected* to return rows in
    ``[0, 1, ..., num_envs-1]`` order.  We do not rely on that: we
    explicitly reorder so the runner is correct under any permutation
    (e.g. async mode, future API changes, partial-batch returns).
    """
    if not isinstance(info, dict) or "env_id" not in info:
        raise RuntimeError(
            f"envpool {where} did not return info['env_id']; cannot align rows safely"
        )
    eid = np.asarray(info["env_id"])
    expected_sorted = np.sort(np.asarray(expected_ids))
    if not np.array_equal(np.sort(eid), expected_sorted):
        raise RuntimeError(
            f"envpool {where} returned env_id {eid.tolist()}, "
            f"expected the set {expected_sorted.tolist()}"
        )
    if np.array_equal(eid, expected_ids):
        return arrays  # already aligned (the common synchronous case)
    # Permutation: build a row-permutation ``perm`` such that
    # ``arr[perm][k]`` is the row for ``expected_ids[k]``.
    pos = {int(e): k for k, e in enumerate(eid)}
    perm = np.array([pos[int(e)] for e in expected_ids], dtype=np.int64)
    return tuple(a[perm] for a in arrays)


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
    master_seed: int = DEFAULT_ENVPOOL_MASTER_SEED,
    pool_size: int | None = None,
    on_complete: Callable[[dict], None] | None = None,
) -> Iterator[dict]:
    """Run one batch of N agents against an envpool batch.

    Each label ``s`` is bound to envpool sub-env index ``s``; ``pool_size``
    must be ``> max(seeds)``.  Yields the run dict for each agent as soon
    as it finishes ``n_episodes`` (so a mid-batch crash preserves
    completed work via ``on_complete``).

    Parameters
    ----------
    pool_size:
        Number of sub-envs in the envpool.  Defaults to ``max(seeds)+1``.
        Sub-envs at indices not in ``seeds`` are stepped with action 0
        and their results discarded.
    on_complete:
        Optional callback invoked with each run dict the moment its
        agent finishes.  Use this to persist results incrementally.

    Yields
    ------
    dict
        One run dict per agent, in completion order (not necessarily the
        order of ``labels``).
    """
    n = len(labels)
    if not (len(seeds) == n and len(agent_kwargs_list) == n):
        raise ValueError(
            f"length mismatch: labels={n}, seeds={len(seeds)}, "
            f"agent_kwargs_list={len(agent_kwargs_list)}"
        )
    if n == 0:
        raise ValueError("empty batch")
    if any(s < 0 for s in seeds):
        raise ValueError(f"envpool path requires non-negative seeds; got {seeds}")
    if len(set(seeds)) != n:
        raise ValueError(f"duplicate seeds in batch: {seeds}")

    if pool_size is None:
        pool_size = max(seeds) + 1
    if pool_size <= max(seeds):
        raise ValueError(f"pool_size={pool_size} must exceed max seed {max(seeds)}")
    # Sparse-seed footgun: e.g. ``--seeds 0 100`` would create a 101-env
    # pool to run 2 labels.  Warn loudly so this is intentional, not
    # accidental.
    if pool_size > 4 * n:
        print(
            f"  WARNING: sparse seeds inflate envpool size: pool_size={pool_size}"
            f" for only {n} labels (max seed={max(seeds)}). Consider remapping seeds.",
            flush=True,
        )

    # --- Slot mapping: agent i -> sub-env index slot_of_agent[i] = seeds[i] ---
    slot_of_agent: list[int] = list(seeds)

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

    # --- Sequential per-agent warmup against private gym envs ---
    # NEFFQIAgent.warmup calls env.reset() without a seed argument, so
    # the env's RNG would otherwise seed itself from system entropy.
    # To preserve label-invariance across batches we pre-seed the env
    # with the agent's seed; subsequent reset() calls inside warmup
    # continue deterministically from that RNG state.  This is a
    # *different* deterministic warmup than the gym path's (which is
    # non-deterministic in its own subprocess); statistical parity is
    # what we check, not byte-identical trajectories.
    for i in range(n):
        warmup_env = gym.make(env_name)
        warmup_env.reset(seed=int(seeds[i]))
        agents[i].warmup(env=warmup_env, n_episodes=20)
        warmup_env.close()

    # --- Build envpool batch ---
    pool = envpool.make(
        env_name,
        env_type="gymnasium",
        num_envs=pool_size,
        seed=int(master_seed),
        max_episode_steps=max_episode_steps,
    )

    rewards_per: list[list[float]] = [[] for _ in range(n)]
    eval_history_per: list[list[dict]] = [[] for _ in range(n)]
    ep_counts = [0] * n
    cur_ep_reward = [0.0] * n
    finished = [False] * n
    yielded = [False] * n
    elapsed_per: list[float | None] = [None] * n

    # Per-agent "last obs the agent acted on" — never indexed by slot at
    # decision time, so explicit resets inside the step loop don't
    # corrupt next-step inputs.
    batch_obs_full, info_r = pool.reset()
    all_ids = np.arange(pool_size, dtype=np.int64)
    (batch_obs_full,) = _reorder_by_env_id(
        info_r, (batch_obs_full,), all_ids, where="initial reset"
    )
    last_obs_per: list[np.ndarray] = [batch_obs_full[slot_of_agent[i]].copy() for i in range(n)]

    t_start = time.time()

    try:
        while not all(finished):
            # Build action vector over ALL slots; idle/finished slots get 0.
            actions = np.zeros(pool_size, dtype=np.int32)
            for i in range(n):
                if not finished[i]:
                    actions[slot_of_agent[i]] = agents[i].select_action(last_obs_per[i])

            next_obs_full, rewards, term, trunc, info_s = pool.step(actions)
            next_obs_full, rewards, term, trunc = _reorder_by_env_id(
                info_s,
                (next_obs_full, rewards, term, trunc),
                all_ids,
                where="step",
            )

            to_reset_slots: list[int] = []
            just_finished: list[int] = []
            for i in range(n):
                if finished[i]:
                    continue
                slot = slot_of_agent[i]
                done = bool(term[slot] or trunc[slot])
                agents[i].update(
                    last_obs_per[i],
                    int(actions[slot]),
                    float(rewards[slot]),
                    next_obs_full[slot],
                    done,
                )
                cur_ep_reward[i] += float(rewards[slot])
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
                        just_finished.append(i)
                    else:
                        to_reset_slots.append(slot)

                # Update last_obs_per BEFORE explicit-reset overwrite below.
                last_obs_per[i] = next_obs_full[slot].copy()

            if to_reset_slots:
                # Override envpool's auto-reset state with an explicit reset
                # so the next ``select_action`` sees a true first-of-episode
                # obs (matches gym's env.reset() semantics for the agent's
                # decision input, even if the per-episode RNG seed differs).
                reset_ids = np.array(to_reset_slots, dtype=np.int32)
                reset_obs, reset_info = pool.reset(env_id=reset_ids)
                # Defensive: realign returned obs by env_id so row k
                # corresponds to ``to_reset_slots[k]``, not request order.
                (reset_obs,) = _reorder_by_env_id(
                    reset_info, (reset_obs,), reset_ids.astype(np.int64), where="subset reset"
                )
                slot_to_reset_idx = {slot: k for k, slot in enumerate(to_reset_slots)}
                for i in range(n):
                    if finished[i]:
                        continue
                    slot = slot_of_agent[i]
                    if slot in slot_to_reset_idx:
                        last_obs_per[i] = reset_obs[slot_to_reset_idx[slot]].copy()

            # Yield completed agents eagerly so callers can persist them
            # before any later batch-level failure.
            for i in just_finished:
                if yielded[i]:
                    continue
                run_dict = _build_run_dict(
                    label=labels[i],
                    seed=seeds[i],
                    knob=knobs[i],
                    value=values[i],
                    rewards=rewards_per[i],
                    eval_history=eval_history_per[i],
                    elapsed=float(elapsed_per[i] or 0.0),
                    config=clean_kwargs[i],
                )
                yielded[i] = True
                if on_complete is not None:
                    on_complete(run_dict)
                yield run_dict
    finally:
        pool.close()


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
        help=(
            "Output JSON. Defaults to results/rl/<knob>_lunarlander_envpool.json "
            "to avoid clobbering the gym path's results."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="Per-episode truncation (matches gym v3 default of 1000)",
    )
    parser.add_argument(
        "--envpool-master-seed",
        type=int,
        default=DEFAULT_ENVPOOL_MASTER_SEED,
        help=(
            "Master seed for envpool.make().  Combined with each label's "
            "seed s -> sub-env index s, this fully determines the per-label "
            "training trajectory.  Change at your own risk: results in the "
            "output JSON were produced with the master seed implied by "
            "previous runs."
        ),
    )
    args = parser.parse_args()

    base = load_base_config(args.base_config)
    parsed_values = [parse_value(args.knob, v) for v in args.values]
    out_path = args.output or f"results/rl/{args.knob}_lunarlander_envpool.json"

    if any(s < 0 for s in args.seeds):
        parser.error(f"envpool path requires non-negative integer seeds; got {args.seeds}")
    pool_size = max(args.seeds) + 1

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
        print(
            f"  pool_size={pool_size}  master_seed={args.envpool_master_seed}  "
            f"(label s pinned to sub-env index s)"
        )
        print(f"  base_config from: {args.base_config}")
        print(f"  output: {out_path}")
        print(f"{'=' * 70}\n")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        for bi, (labels, seeds, kwargs_list) in enumerate(batches):
            print(f"  [batch {bi + 1}/{len(batches)}] running {len(labels)} agents: {labels}")
            t_batch = time.time()
            saved_in_batch: set[str] = set()

            def _save(r: dict, _saved=saved_in_batch, _out=out_path) -> None:
                existing[r["label"]] = r
                _atomic_write_json(_out, existing)
                _saved.add(r["label"])
                print(f"  → saved {r['label']} to {_out}", flush=True)

            try:
                for _ in run_envpool_batch(
                    env_name=args.env,
                    n_episodes=args.episodes,
                    labels=labels,
                    seeds=seeds,
                    agent_kwargs_list=kwargs_list,
                    max_episode_steps=args.max_episode_steps,
                    master_seed=args.envpool_master_seed,
                    pool_size=pool_size,
                    on_complete=_save,
                ):
                    pass  # _save already persisted; iterator just drains.
            except Exception as e:
                print(f"  [batch {bi + 1}] FAILED: {e}", flush=True)
                # Record errors only for labels that didn't already finish.
                for label, seed in zip(labels, seeds):
                    if label in saved_in_batch:
                        continue
                    existing[label] = {"label": label, "error": str(e), "seed": seed}
                _atomic_write_json(out_path, existing)
                continue

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
