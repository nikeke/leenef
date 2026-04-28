"""Sparse-reward benchmark driver for NEF-FQI vs NEF-SF (+ optional D2 novelty).

Wraps the existing :mod:`benchmarks.run_rl` training loop around the new
:class:`leenef.rl_sf.NEFSFAgent`, so a single multi-seed call produces
comparable JSON rows for ``nef_fqi``, ``nef_sf``, and ``nef_sf_novelty``
on a target environment.  Reuses :mod:`benchmarks.rl_repro`'s seed
policy (warmup_seed / training_seed / EVAL_SEED_BASE) verbatim.

Usage::

    python -u benchmarks/run_rl_sparse.py \\
        --env MountainCar-v0 --episodes 300 --seeds 0 1 2 \\
        --variants nef_fqi nef_sf nef_sf_novelty \\
        --output results/rl_sparse/mountaincar.json

The base configuration is loaded from ``results/rl_clean/base_config.json``
on every run so it tracks the rl_clean sweep adoption decisions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from benchmarks.rl_repro import (
    BASE_CONFIG_PATH,
    EVAL_SEED_BASE,
    code_version,
    reset_env,
    result_metadata,
    robust_eval_from_history,
    seed_policy,
    set_rl_seed,
    training_seed,
    warmup_seed,
)
from leenef.rl import NEFFQIAgent
from leenef.rl_diag import (
    SparseRewardDiagnostics,
    leverage_stats,
    solve_conditioning,
    sparse_metric_summary,
)
from leenef.rl_sf import NEFHindsightSFAgent, NEFSFAgent

VARIANTS = ("nef_fqi", "nef_sf", "nef_sf_novelty", "nef_sf_hindsight")


def _load_base_config() -> dict:
    return json.loads(BASE_CONFIG_PATH.read_text())


def _build_agent(
    variant: str,
    base_config: dict,
    *,
    d_obs: int,
    n_actions: int,
    seed: int,
    novelty_beta: float,
    goal_buffer_size: int,
    goal_horizon: int,
    goal_warmup_solves: int,
    goal_mode: str,
    goal_random_prob: float,
    goal_action_mode: str,
):
    """Construct an agent of the given variant from the rl_clean base config.

    NEF-SF reuses the FQI hyperparameters that map to its own surface
    (n_neurons, gamma, alpha, n_step, forget_factor, solve_every,
    epsilon schedule).  Mode-specific knobs (target_mode, broadness,
    fqi_iters, recenter_interval, tau) are dropped silently for SF.
    """
    if variant == "nef_fqi":
        cfg = dict(base_config)
        return NEFFQIAgent(d_obs=d_obs, n_actions=n_actions, seed=seed, **cfg)
    if variant in ("nef_sf", "nef_sf_novelty", "nef_sf_hindsight"):
        bonus = float(novelty_beta) if variant == "nef_sf_novelty" else 0.0
        common = dict(
            d_obs=d_obs,
            n_actions=n_actions,
            n_neurons=int(base_config["n_neurons"]),
            gamma=float(base_config["gamma"]),
            alpha=float(base_config["alpha"]),
            n_step=int(base_config["n_step"]),
            forget_factor=float(base_config["forget_factor"]),
            epsilon_start=float(base_config["epsilon_start"]),
            epsilon_end=float(base_config["epsilon_end"]),
            epsilon_decay=int(base_config["epsilon_decay"]),
            solve_every=int(base_config["solve_every"]),
            warmup_episodes=0,
            novelty_beta=bonus,
            seed=seed,
        )
        if variant == "nef_sf_hindsight":
            return NEFHindsightSFAgent(
                **common,
                goal_buffer_size=int(goal_buffer_size),
                goal_horizon=int(goal_horizon),
                goal_warmup_solves=int(goal_warmup_solves),
                goal_mode=goal_mode,
                goal_random_prob=float(goal_random_prob),
                goal_action_mode=goal_action_mode,
            )
        return NEFSFAgent(**common)
    raise ValueError(f"Unknown variant {variant!r}, expected one of {VARIANTS}")


def _evaluate(agent, env_name: str, n_episodes: int, seed: int) -> float:
    env = gym.make(env_name)
    returns: list[float] = []
    for i in range(n_episodes):
        obs, _ = reset_env(env, seed=seed + i)
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                q = agent.q_values(obs)
                action = int(q.argmax().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = bool(terminated or truncated)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def _agent_diagnostics(agent) -> dict[str, Any]:
    if hasattr(agent, "successor"):
        ata = agent.successor._ata
        feats = agent.successor.features
    else:
        ata = getattr(agent, "_ata", None)
        feats = agent.features
    if ata is None:
        return {}
    n_actions = ata.shape[0]
    cond = [solve_conditioning(ata[a], alpha=agent.alpha) for a in range(n_actions)]
    # Sample 256 random states from the encoder normalization range as a leverage probe.
    rng = np.random.default_rng(0)
    probe = torch.from_numpy(rng.normal(size=(256, feats.d_in)).astype(np.float32))
    with torch.no_grad():
        phi = feats.encode(probe).double()
    lev = leverage_stats(ata.sum(dim=0), alpha=agent.alpha, phi_samples=phi)
    return {
        "per_action_conditioning": [
            {k: float(v) for k, v in c.items() if not isinstance(v, torch.Tensor)} for c in cond
        ],
        "leverage": {k: float(v) for k, v in lev.items() if not isinstance(v, torch.Tensor)},
    }


def _train(
    variant: str,
    env_name: str,
    base_config: dict,
    seed: int,
    *,
    n_episodes: int,
    eval_every: int,
    eval_episodes: int,
    novelty_beta: float,
    warmup_episodes: int,
    goal_buffer_size: int,
    goal_horizon: int,
    goal_warmup_solves: int,
    goal_mode: str,
    goal_random_prob: float,
    goal_action_mode: str,
    verbose: bool,
) -> dict:
    set_rl_seed(seed)
    env = gym.make(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = _build_agent(
        variant,
        base_config,
        d_obs=d_obs,
        n_actions=n_actions,
        seed=seed,
        novelty_beta=novelty_beta,
        goal_buffer_size=goal_buffer_size,
        goal_horizon=goal_horizon,
        goal_warmup_solves=goal_warmup_solves,
        goal_mode=goal_mode,
        goal_random_prob=goal_random_prob,
        goal_action_mode=goal_action_mode,
    )

    warmup_env = gym.make(env_name)
    agent.warmup(env=warmup_env, n_episodes=warmup_episodes, seed=warmup_seed(seed))
    warmup_env.close()

    diag = SparseRewardDiagnostics()
    eval_history: list[dict] = []
    intrinsic_history: list[float] = []
    t_start = time.time()

    for ep in range(n_episodes):
        obs, _ = reset_env(env, seed=training_seed(seed, ep))
        done = False
        term = False
        ep_intrinsic = 0.0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            term = bool(terminated)
            diag.record_step(float(reward), action=int(action))
            agent.update(obs, action, reward, next_obs, done)
            if hasattr(agent, "last_intrinsic_reward"):
                ep_intrinsic += float(agent.last_intrinsic_reward)
            obs = next_obs
        diag.end_episode(terminated=term)
        intrinsic_history.append(ep_intrinsic)
        if hasattr(agent, "end_episode"):
            agent.end_episode()

        if (ep + 1) % eval_every == 0:
            eval_return = _evaluate(agent, env_name, eval_episodes, seed=EVAL_SEED_BASE)
            eval_history.append({"episode": ep + 1, "eval": eval_return})
            if verbose:
                recent = float(np.mean(diag.episode_rewards[-eval_every:]))
                eps = agent._epsilon() if hasattr(agent, "_epsilon") else float("nan")
                print(
                    f"  [{variant}/seed={seed}] ep {ep + 1:4d}  "
                    f"train_avg={recent:8.2f}  eval={eval_return:8.2f}  ε={eps:.3f}",
                    flush=True,
                )

    t_elapsed = time.time() - t_start
    final_eval = _evaluate(agent, env_name, eval_episodes, seed=EVAL_SEED_BASE + 100_000)
    env.close()

    metrics = sparse_metric_summary(eval_history, threshold=0.0)
    return {
        "variant": variant,
        "seed": seed,
        "final_eval": round(final_eval, 4),
        "best_eval": (
            round(max(e["eval"] for e in eval_history), 4) if eval_history else final_eval
        ),
        "robust_eval": robust_eval_from_history(eval_history),
        "eval_history": eval_history,
        "diagnostics": diag.summary(),
        "sparse_metrics": metrics,
        "agent_diagnostics": _agent_diagnostics(agent),
        "intrinsic_total": round(float(sum(intrinsic_history)), 6),
        "intrinsic_history": [round(v, 4) for v in intrinsic_history],
        "time_s": round(t_elapsed, 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--novelty-beta", type=float, default=0.5)
    parser.add_argument("--warmup-episodes", type=int, default=20)
    parser.add_argument("--goal-buffer-size", type=int, default=1024)
    parser.add_argument("--goal-horizon", type=int, default=50)
    parser.add_argument("--goal-warmup-solves", type=int, default=1)
    parser.add_argument(
        "--goal-mode",
        choices=("buffer", "random_direction", "mixed"),
        default="buffer",
        help="hindsight goal sampling: buffer=visited states (default), "
        "random_direction=uniform unit vector in feature space, "
        "mixed=coin-flip with --goal-random-prob.",
    )
    parser.add_argument("--goal-random-prob", type=float, default=0.5)
    parser.add_argument(
        "--goal-action-mode",
        choices=("epsilon", "commit"),
        default="epsilon",
        help="epsilon=ε-greedy, goal-Q only on greedy steps (default); "
        "commit=after warmup take goal-greedy every step, ε resamples goal.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a base_config key (e.g. --config-override n_neurons=2000).",
    )
    args = parser.parse_args(argv)

    base_config = _load_base_config()
    for kv in args.config_override:
        if "=" not in kv:
            raise ValueError(f"--config-override expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in base_config:
            raise KeyError(f"Unknown base_config key {k!r}")
        original = base_config[k]
        if isinstance(original, bool):
            base_config[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(original, int) and not isinstance(original, bool):
            base_config[k] = int(v)
        elif isinstance(original, float):
            base_config[k] = float(v)
        elif original is None:
            base_config[k] = None if v.lower() in ("none", "null") else v
        else:
            base_config[k] = v
    metadata = result_metadata(
        config=base_config,
        seed=-1,  # multi-seed driver
        run_context={
            "env_name": args.env,
            "n_episodes": args.episodes,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "warmup_episodes": args.warmup_episodes,
            "variants": args.variants,
            "novelty_beta": args.novelty_beta,
            "goal_buffer_size": args.goal_buffer_size,
            "goal_horizon": args.goal_horizon,
            "goal_warmup_solves": args.goal_warmup_solves,
            "goal_mode": args.goal_mode,
            "goal_random_prob": args.goal_random_prob,
            "goal_action_mode": args.goal_action_mode,
        },
    )
    metadata["seed_policy"] = seed_policy()
    metadata["code_version"] = code_version()

    runs: list[dict] = []
    if args.output is not None and args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
            if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                runs = list(existing["runs"])
        except json.JSONDecodeError:
            pass

    completed = {(r.get("variant"), r.get("seed")) for r in runs}

    for variant in args.variants:
        for seed in args.seeds:
            key = (variant, int(seed))
            if key in completed:
                print(f"  [{variant}/seed={seed}] skipping; already in {args.output}", flush=True)
                continue
            result = _train(
                variant=variant,
                env_name=args.env,
                base_config=base_config,
                seed=int(seed),
                n_episodes=args.episodes,
                eval_every=args.eval_every,
                eval_episodes=args.eval_episodes,
                novelty_beta=args.novelty_beta,
                warmup_episodes=args.warmup_episodes,
                goal_buffer_size=args.goal_buffer_size,
                goal_horizon=args.goal_horizon,
                goal_warmup_solves=args.goal_warmup_solves,
                goal_mode=args.goal_mode,
                goal_random_prob=args.goal_random_prob,
                goal_action_mode=args.goal_action_mode,
                verbose=not args.quiet,
            )
            runs.append(result)
            # Flush after every run (rl_clean / colab-suite policy).
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({**metadata, "runs": runs}, indent=2, default=str) + "\n"
                )

    summary = {**metadata, "runs": runs}
    if args.output is None:
        print(json.dumps(summary, indent=2, default=str))
    else:
        args.output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        if not args.quiet:
            print(f"\nWrote {len(runs)} runs to {args.output}", flush=True)

    return summary


if __name__ == "__main__":
    main()
