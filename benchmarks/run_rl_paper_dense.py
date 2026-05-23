#!/usr/bin/env python3
"""Dense-reward RL panels for SSUP Paper III.

The runner writes resumable JSON rows under ``results/rl_paper/dense``.
It uses the same seed policy as the prior dense-control sweep series but
keeps this paper-facing replication namespace separate from tuning history.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.rl_repro import (
    EVAL_SEED_BASE,
    code_version,
    reset_env,
    robust_eval_from_history,
    seed_policy,
    set_rl_seed,
    training_seed,
    warmup_seed,
)
from leenef.rl import NEFFQIAgent, NEFFQIEnsemble

RESULT_NAMESPACE = "rl_paper_dense"
RUNNER_VERSION = "rl-paper-dense-v1"
RESULTS_DIR = Path("results") / "rl_paper" / "dense"

_TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1")))
torch.set_num_threads(max(1, _TORCH_THREADS))

FINAL_LUNAR_CONFIG = {
    "n_neurons": 6000,
    "gamma": 0.98,
    "alpha": 0.1,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 500,
    "solve_every": 10,
    "target_mode": "nstep",
    "n_step": 100,
    "forget_factor": 0.999,
    "fqi_iters": 1,
    "tau": 0.3,
    "broadness": 0.0,
    "recenter_interval": None,
}


class DQNAgent:
    """Small DQN baseline with replay and a target network."""

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        *,
        hidden: int = 256,
        gamma: float = 0.98,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update: int = 10,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(epsilon_decay)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.q_net = nn.Sequential(
            nn.Linear(d_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.target_net = nn.Sequential(
            nn.Linear(d_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._buffer_size = int(buffer_size)
        self._buffer_pos = 0
        self._episode = 0

    def _epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self._episode / self.epsilon_decay
        )

    def select_action(self, obs: np.ndarray) -> int:
        if self.rng.random() < self._epsilon():
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.q_net(torch.from_numpy(obs).float().unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def q_values(self, obs: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.q_net(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        transition = (obs, int(action), float(reward), next_obs, bool(done))
        if len(self.buffer) < self._buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self._buffer_pos] = transition
            self._buffer_pos = (self._buffer_pos + 1) % self._buffer_size

        if len(self.buffer) < self.batch_size:
            return

        idx = self.rng.integers(len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idx]
        states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q
        q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def end_episode(self) -> None:
        self._episode += 1
        if self._episode % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


def _metadata(config: dict[str, Any], run_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_namespace": RESULT_NAMESPACE,
        "runner_version": RUNNER_VERSION,
        "config": config,
        "seed_policy": seed_policy(),
        "code_version": code_version(),
        "run_context": run_context,
    }


def _make_env(env_name: str):
    return gym.make(env_name)


def _env_shape(env_name: str) -> tuple[int, int]:
    env = _make_env(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()
    return int(d_obs), int(n_actions)


def evaluate(agent, env_name: str, *, n_episodes: int, seed: int) -> float:
    env = _make_env(env_name)
    returns: list[float] = []
    for i in range(n_episodes):
        obs, _ = reset_env(env, seed=seed + i)
        done = False
        total = 0.0
        while not done:
            q = agent.q_values(obs)
            action = int(q.argmax().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
    env.close()
    return round(float(np.mean(returns)), 1)


def _warmup_if_needed(agent, env_name: str, seed: int, n_episodes: int) -> None:
    if not hasattr(agent, "warmup") or n_episodes <= 0:
        return
    env = _make_env(env_name)
    agent.warmup(env=env, n_episodes=n_episodes, seed=warmup_seed(seed))
    env.close()


def train_agent(
    *,
    label: str,
    env_name: str,
    agent,
    seed: int,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    warmup_episodes: int,
) -> dict[str, Any]:
    set_rl_seed(seed)
    _warmup_if_needed(agent, env_name, seed, warmup_episodes)

    env = _make_env(env_name)
    rewards: list[float] = []
    eval_history: list[dict[str, Any]] = []
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = reset_env(env, seed=training_seed(seed, ep))
        done = False
        total = 0.0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total += float(reward)
        agent.end_episode()
        rewards.append(total)

        if (ep + 1) % eval_every == 0:
            eval_return = evaluate(
                agent,
                env_name,
                n_episodes=eval_episodes,
                seed=EVAL_SEED_BASE,
            )
            eval_history.append({"episode": ep + 1, "eval": eval_return})
            recent = float(np.mean(rewards[-eval_every:]))
            eps = agent._epsilon() if hasattr(agent, "_epsilon") else float("nan")
            print(
                f"  [{label}/seed={seed}] ep={ep + 1:4d} "
                f"train_avg={recent:8.1f} eval={eval_return:8.1f} eps={eps:.3f}",
                flush=True,
            )

    elapsed = time.time() - t0
    final_eval = evaluate(
        agent,
        env_name,
        n_episodes=eval_episodes,
        seed=EVAL_SEED_BASE + 100_000,
    )
    env.close()

    arr = np.array(rewards)
    window = min(50, len(arr))
    rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
    return {
        "label": label,
        "seed": seed,
        "best_50": round(float(rolling.max()), 1),
        "final_50": round(float(rolling[-1]), 1),
        "best_eval": max((e["eval"] for e in eval_history), default=final_eval),
        "final_eval": final_eval,
        "robust_eval": robust_eval_from_history(eval_history),
        "eval_history": eval_history,
        "time_s": round(elapsed, 1),
    }


def _build_nef(env_name: str, config: dict[str, Any], seed: int):
    d_obs, n_actions = _env_shape(env_name)
    return NEFFQIAgent(d_obs=d_obs, n_actions=n_actions, seed=seed, **config)


def _build_dqn(env_name: str, config: dict[str, Any], seed: int):
    d_obs, n_actions = _env_shape(env_name)
    return DQNAgent(d_obs=d_obs, n_actions=n_actions, seed=seed, **config)


def _build_ensemble(env_name: str, config: dict[str, Any], seed: int):
    d_obs, n_actions = _env_shape(env_name)
    kwargs = dict(config)
    n_members = int(kwargs.pop("n_members"))
    explore_strategy = str(kwargs.pop("explore_strategy"))
    return NEFFQIEnsemble(
        d_obs=d_obs,
        n_actions=n_actions,
        n_members=n_members,
        explore_strategy=explore_strategy,
        base_seed=seed * 100,
        **kwargs,
    )


def phase_specs(phase: str, *, episodes: int | None = None) -> list[dict[str, Any]]:
    final = dict(FINAL_LUNAR_CONFIG)
    phases: dict[str, list[dict[str, Any]]] = {
        "lunar_final": [
            {
                "label": "lunar_nef_final",
                "env_name": "LunarLander-v3",
                "kind": "nef",
                "episodes": episodes or 1000,
                "eval_every": 50,
                "eval_episodes": 20,
                "warmup_episodes": 20,
                "config": final,
            }
        ],
        "lunar_dqn": [
            {
                "label": "lunar_dqn_256x2",
                "env_name": "LunarLander-v3",
                "kind": "dqn",
                "episodes": episodes or 1000,
                "eval_every": 50,
                "eval_episodes": 20,
                "warmup_episodes": 0,
                "config": {
                    "hidden": 256,
                    "gamma": 0.98,
                    "lr": 1e-3,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.01,
                    "epsilon_decay": 500,
                    "target_update": 10,
                },
            }
        ],
        "bootstrap_lunar": [
            {
                "label": "lunar_nstep_no_bootstrap",
                "env_name": "LunarLander-v3",
                "kind": "nef",
                "episodes": episodes or 500,
                "eval_every": 50,
                "eval_episodes": 20,
                "warmup_episodes": 20,
                "config": {**final, "n_neurons": 4000, "n_step": 100},
            },
            {
                "label": "lunar_mc_no_bootstrap",
                "env_name": "LunarLander-v3",
                "kind": "nef",
                "episodes": episodes or 500,
                "eval_every": 50,
                "eval_episodes": 20,
                "warmup_episodes": 20,
                "config": {**final, "n_neurons": 4000, "target_mode": "mc"},
            },
            {
                "label": "lunar_td_bootstrap",
                "env_name": "LunarLander-v3",
                "kind": "nef",
                "episodes": episodes or 500,
                "eval_every": 50,
                "eval_episodes": 20,
                "warmup_episodes": 20,
                "config": {
                    **final,
                    "n_neurons": 4000,
                    "target_mode": "td",
                    "forget_factor": None,
                    "buffer_size": 50_000,
                    "solve_batch": 20_000,
                    "fqi_iters": 1,
                },
            },
        ],
        "cartpole_sanity": [
            {
                "label": "cartpole_nef_mc",
                "env_name": "CartPole-v1",
                "kind": "nef",
                "episodes": episodes or 500,
                "eval_every": 50,
                "eval_episodes": 20,
                "warmup_episodes": 20,
                "config": {
                    "n_neurons": 2000,
                    "gamma": 0.99,
                    "alpha": 0.1,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.01,
                    "epsilon_decay": 150,
                    "solve_every": 5,
                    "target_mode": "mc",
                    "forget_factor": 0.999,
                    "fqi_iters": 1,
                    "tau": 0.3,
                    "broadness": 0.0,
                    "recenter_interval": None,
                },
            }
        ],
        "ensemble_lunar": [
            {
                "label": "lunar_ensemble_thompson3",
                "env_name": "LunarLander-v3",
                "kind": "ensemble",
                "episodes": episodes or 1000,
                "eval_every": 50,
                "eval_episodes": 20,
                "warmup_episodes": 20,
                "config": {**final, "n_members": 3, "explore_strategy": "thompson"},
            },
            {
                "label": "lunar_ensemble_voting3",
                "env_name": "LunarLander-v3",
                "kind": "ensemble",
                "episodes": episodes or 1000,
                "eval_every": 50,
                "eval_episodes": 20,
                "warmup_episodes": 20,
                "config": {**final, "n_members": 3, "explore_strategy": "voting"},
            },
        ],
    }
    if phase not in phases:
        raise KeyError(f"Unknown phase {phase!r}; choose one of {sorted(phases)}")
    return phases[phase]


def _build_agent(spec: dict[str, Any], seed: int):
    kind = spec["kind"]
    if kind == "nef":
        return _build_nef(spec["env_name"], spec["config"], seed)
    if kind == "dqn":
        return _build_dqn(spec["env_name"], spec["config"], seed)
    if kind == "ensemble":
        return _build_ensemble(spec["env_name"], spec["config"], seed)
    raise ValueError(f"Unknown kind {kind!r}")


def _run_one(args) -> dict[str, Any]:
    phase, spec, seed = args
    try:
        label = spec["label"]
        agent = _build_agent(spec, seed)
        result = train_agent(
            label=label,
            env_name=spec["env_name"],
            agent=agent,
            seed=seed,
            episodes=int(spec["episodes"]),
            eval_every=int(spec["eval_every"]),
            eval_episodes=int(spec["eval_episodes"]),
            warmup_episodes=int(spec["warmup_episodes"]),
        )
        result.update(
            {
                "phase": phase,
                "kind": spec["kind"],
                "env_name": spec["env_name"],
                "config": spec["config"],
                "run_context": _run_context(spec),
                "seed_policy": seed_policy(),
                "code_version": code_version(),
                "runner_version": RUNNER_VERSION,
                "result_namespace": RESULT_NAMESPACE,
            }
        )
        return result
    except Exception as exc:
        return {
            "phase": phase,
            "label": spec.get("label"),
            "seed": seed,
            "error": repr(exc),
            "runner_version": RUNNER_VERSION,
            "result_namespace": RESULT_NAMESPACE,
            "code_version": code_version(),
        }


def _result_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("label")), int(row.get("seed", -1))


def _run_context(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": spec["episodes"],
        "eval_every": spec["eval_every"],
        "eval_episodes": spec["eval_episodes"],
        "warmup_episodes": spec["warmup_episodes"],
    }


def _can_reuse(row: dict[str, Any], spec: dict[str, Any], seed: int) -> bool:
    return (
        "error" not in row
        and row.get("runner_version") == RUNNER_VERSION
        and row.get("result_namespace") == RESULT_NAMESPACE
        and row.get("label") == spec["label"]
        and row.get("seed") == seed
        and row.get("config") == spec["config"]
        and row.get("run_context") == _run_context(spec)
        and row.get("seed_policy") == seed_policy()
        and row.get("code_version") == code_version()
    )


def _write_output(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({**metadata, "runs": rows}, indent=2, default=str) + "\n")
    tmp.replace(path)


def _summarize(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if "error" in row:
            continue
        grouped.setdefault(str(row["label"]), []).append(row)
    for label, rs in sorted(grouped.items()):
        vals = np.array([r["robust_eval"] for r in rs if r.get("robust_eval") is not None])
        finals = np.array([r["final_eval"] for r in rs if r.get("final_eval") is not None])
        if vals.size:
            se = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            fse = float(finals.std(ddof=1) / np.sqrt(finals.size)) if finals.size > 1 else 0.0
            print(
                f"  {label:<30s} n={vals.size:2d} "
                f"robust={vals.mean():8.1f} ± {se:5.1f} "
                f"final={finals.mean():8.1f} ± {fse:5.1f}",
                flush=True,
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    specs = phase_specs(args.phase, episodes=args.episodes)
    output = args.output or (RESULTS_DIR / f"{args.phase}.json")
    metadata = _metadata(
        {"phase": args.phase, "seeds": args.seeds, "specs": specs},
        {"phase": args.phase, "parallel": args.parallel},
    )

    rows: list[dict[str, Any]] = []
    if output.exists():
        existing = json.loads(output.read_text())
        if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
            rows = list(existing["runs"])

    existing_by_key = {_result_key(row): row for row in rows}
    tasks = []
    for spec in specs:
        for seed in args.seeds:
            row = existing_by_key.get((spec["label"], seed))
            if row is not None and _can_reuse(row, spec, seed):
                print(f"  skipping {spec['label']} seed={seed}", flush=True)
                continue
            tasks.append((args.phase, spec, seed))

    print(
        f"Phase {args.phase}: {len(tasks)} pending of "
        f"{len(specs) * len(args.seeds)} planned rows; output={output}",
        flush=True,
    )
    if tasks:
        if args.parallel > 1:
            with Pool(args.parallel) as pool:
                iterator = pool.imap_unordered(_run_one, tasks)
                for row in iterator:
                    existing_by_key[_result_key(row)] = row
                    rows = list(existing_by_key.values())
                    _write_output(output, metadata, rows)
                    print(f"  saved {row.get('label')} seed={row.get('seed')}", flush=True)
        else:
            for task in tasks:
                row = _run_one(task)
                existing_by_key[_result_key(row)] = row
                rows = list(existing_by_key.values())
                _write_output(output, metadata, rows)
                print(f"  saved {row.get('label')} seed={row.get('seed')}", flush=True)

    rows = list(existing_by_key.values())
    _write_output(output, metadata, rows)
    _summarize(rows)
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"Encountered {len(errors)} error row(s); see {output}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
