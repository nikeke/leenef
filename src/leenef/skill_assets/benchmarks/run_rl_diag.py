"""Sparse-reward toy-chain diagnostics harness.

Runs both the plain :class:`NEFAVRAgent` and the new
:class:`NEFSFAgent` on the deterministic :class:`SparseChainEnv` while
logging the diagnostic surface from :mod:`leenef.rl_diag`.  Cheap (a
few seconds) — intended as a smoke test for the diagnostic + SF stack
before we point them at MountainCar / Acrobot.

Usage:
    python -u benchmarks/run_rl_diag.py --n-states 7 --episodes 200

Outputs a small JSON summary to stdout (and optionally to ``--output``)
so the harness composes with sweep drivers without polluting result
namespaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from leenef.rl import NEFAVRAgent
from leenef.rl_diag import (
    SparseChainEnv,
    SparseRewardDiagnostics,
    leverage_stats,
    solve_conditioning,
    sparse_metric_summary,
)
from leenef.rl_sf import NEFSFAgent


def _run_agent(agent_name: str, agent, env: SparseChainEnv, n_episodes: int, seed: int):
    diag = SparseRewardDiagnostics()
    has_end_episode = hasattr(agent, "end_episode")
    eval_returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        term = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            diag.record_step(float(reward), action=int(action))
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
        diag.end_episode(terminated=bool(term))
        if has_end_episode:
            agent.end_episode()
        # Periodic eval by greedy rollout (no learning).
        if (ep + 1) % max(1, n_episodes // 20) == 0:
            eval_returns.append(_greedy_rollout(agent, env, seed=10_000 + ep))
    return diag, eval_returns


def _greedy_rollout(agent, env: SparseChainEnv, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    done = False
    ret = 0.0
    while not done:
        with torch.no_grad():
            q = agent.q_values(obs)
            action = int(q.argmax().item())
        obs, reward, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        ret += float(reward)
    return ret


def _agent_stats(agent) -> dict[str, Any]:
    """Extract the AᵀA-derived diagnostics from either agent type."""
    if hasattr(agent, "successor"):
        ata_per_action = agent.successor._ata
    else:
        ata_per_action = agent._ata if hasattr(agent, "_ata") else None
    if ata_per_action is None:
        return {}
    n_actions = ata_per_action.shape[0]
    grid = torch.linspace(0.0, 1.0, steps=21).unsqueeze(1).float()
    if hasattr(agent, "successor"):
        with torch.no_grad():
            phi = agent.successor.features.encode(grid).double()
    else:
        with torch.no_grad():
            phi = agent.features.encode(grid).double()
    ata_sum = ata_per_action.sum(dim=0)
    lev = leverage_stats(ata_sum, alpha=agent.alpha, phi_samples=phi)
    cond = []
    for a in range(n_actions):
        cond.append(solve_conditioning(ata_per_action[a], alpha=agent.alpha))
    return {
        "per_action_conditioning": [
            {k: float(v) for k, v in c.items() if not isinstance(v, torch.Tensor)} for c in cond
        ],
        "leverage": {k: float(v) for k, v in lev.items() if not isinstance(v, torch.Tensor)},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-states", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--n-neurons", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--n-step", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--novelty-beta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    summary: dict[str, Any] = {"config": vars(args).copy()}
    summary["config"]["output"] = str(args.output) if args.output else None

    env = SparseChainEnv(n_states=args.n_states, max_steps=args.n_states * 4)

    # NEF-AVR baseline
    avr = NEFAVRAgent(
        d_obs=1,
        n_actions=env.action_space.n,
        n_neurons=args.n_neurons,
        gamma=args.gamma,
        alpha=args.alpha,
        n_step=args.n_step,
        target_mode="nstep",
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=max(1, args.episodes // 4),
        solve_every=10,
        seed=args.seed,
    )
    avr_diag, avr_eval = _run_agent("nef_avr", avr, env, args.episodes, seed=args.seed)
    avr_eval_history = [{"episode": i, "eval": v} for i, v in enumerate(avr_eval)]
    summary["nef_avr"] = {
        "diagnostics": avr_diag.summary(),
        "metrics": sparse_metric_summary(avr_eval_history, threshold=0.5),
        "eval_returns": avr_eval,
        "agent_stats": _agent_stats(avr),
    }

    # SF (+ optional novelty)
    sf = NEFSFAgent(
        d_obs=1,
        n_actions=env.action_space.n,
        n_neurons=args.n_neurons,
        gamma=args.gamma,
        alpha=args.alpha,
        n_step=args.n_step,
        novelty_beta=args.novelty_beta,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=max(1, args.episodes // 4),
        solve_every=10,
        warmup_episodes=max(5, args.episodes // 20),
        seed=args.seed,
    )
    sf_diag, sf_eval = _run_agent("nef_sf", sf, env, args.episodes, seed=args.seed)
    sf_eval_history = [{"episode": i, "eval": v} for i, v in enumerate(sf_eval)]
    summary["nef_sf"] = {
        "diagnostics": sf_diag.summary(),
        "metrics": sparse_metric_summary(sf_eval_history, threshold=0.5),
        "eval_returns": sf_eval,
        "agent_stats": _agent_stats(sf),
    }

    # Closed-form optimal Q for reference.
    q_star = env.optimal_q().tolist()
    summary["optimal_q"] = q_star

    text = json.dumps(summary, indent=2, default=lambda o: str(o))
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
