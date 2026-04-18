"""NEF-based reinforcement learning benchmark.

Compares NEF-FQI (Fitted Q-Iteration with analytical solve and random
features) against DQN on classic control tasks.

The core idea: replace DQN's gradient updates with NEF's analytical
least-squares solve.  Both use a replay buffer, but NEF-FQI solves
for Q-weights in one shot via (ΦᵀΦ + αI)⁻¹ ΦᵀY — no learning rate,
no gradient instability, no target network needed.

Usage:
    python benchmarks/run_rl.py                          # all envs
    python benchmarks/run_rl.py --env CartPole-v1        # single env
    python benchmarks/run_rl.py --env CartPole-v1 --episodes 500
    python benchmarks/run_rl.py --skip-dqn               # NEF only
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from leenef.rl import NEFFQIAgent

# ---------------------------------------------------------------------------
# DQN Agent (baseline)
# ---------------------------------------------------------------------------


class DQNAgent:
    """Standard DQN with experience replay and target network."""

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        hidden: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update: int = 10,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
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

        self.buffer = deque(maxlen=buffer_size)
        self._episode = 0
        self._total_steps = 0

    def _epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self._episode / self.epsilon_decay
        )

    def select_action(self, obs: np.ndarray) -> int:
        if self.rng.random() < self._epsilon():
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            s = torch.from_numpy(obs).float().unsqueeze(0)
            q = self.q_net(s)
            return int(q.argmax(dim=1).item())

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))
        self._total_steps += 1

        if len(self.buffer) < self.batch_size:
            return

        indices = self.rng.integers(len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in indices]
        states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1 - dones) * next_q

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def end_episode(self) -> None:
        self._episode += 1
        if self._episode % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ---------------------------------------------------------------------------
# Evaluation and training loops
# ---------------------------------------------------------------------------


def evaluate(
    agent: NEFFQIAgent | DQNAgent,
    env_name: str,
    n_episodes: int = 20,
    seed: int = 42,
) -> float:
    """Evaluate agent greedily over n_episodes, return mean reward."""
    env = gym.make(env_name)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        done = False
        while not done:
            if isinstance(agent, NEFFQIAgent):
                q = agent.q_values(obs)
                action = int(q.argmax().item())
            else:
                with torch.no_grad():
                    s = torch.from_numpy(obs).float().unsqueeze(0)
                    q = agent.q_net(s)
                    action = int(q.argmax(dim=1).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
    env.close()
    return float(np.mean(returns))


def train_agent(
    agent: NEFFQIAgent | DQNAgent,
    env_name: str,
    n_episodes: int = 1000,
    eval_every: int = 50,
    eval_episodes: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Train agent and return results dict."""
    env = gym.make(env_name)
    eval_history = []
    episode_rewards = []

    if isinstance(agent, NEFFQIAgent):
        warmup_env = gym.make(env_name)
        agent.warmup(env=warmup_env, n_episodes=20)
        warmup_env.close()

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
        episode_rewards.append(total_reward)

        if (ep + 1) % eval_every == 0:
            eval_return = evaluate(agent, env_name, eval_episodes, seed=10000)
            eval_history.append({"episode": ep + 1, "eval_return": eval_return})
            if verbose:
                recent = np.mean(episode_rewards[-eval_every:])
                eps = agent._epsilon()
                print(
                    f"  ep {ep + 1:4d}  train_avg={recent:7.1f}  "
                    f"eval={eval_return:7.1f}  ε={eps:.3f}"
                )

    t_elapsed = time.time() - t_start
    env.close()

    final_eval = evaluate(agent, env_name, eval_episodes, seed=99999)

    return {
        "final_eval": final_eval,
        "best_eval": max(e["eval_return"] for e in eval_history) if eval_history else final_eval,
        "eval_history": eval_history,
        "time_s": round(t_elapsed, 2),
        "episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# Environment-specific configurations
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "CartPole-v1": {
        "n_neurons": 2000,
        "episodes": 500,
        "gamma": 0.99,
        "epsilon_decay": 150,
        "solve_every": 5,
        "target_mode": "mc",
        "dqn_hidden": 128,
        "dqn_lr": 1e-3,
        "dqn_epsilon_decay": 200,
        "dqn_target_update": 10,
    },
    "MountainCar-v0": {
        "n_neurons": 4000,
        "episodes": 1000,
        "gamma": 0.99,
        "epsilon_decay": 300,
        "solve_every": 10,
        "target_mode": "mc",
        "dqn_hidden": 128,
        "dqn_lr": 1e-3,
        "dqn_epsilon_decay": 400,
        "dqn_target_update": 10,
    },
    "Acrobot-v1": {
        "n_neurons": 4000,
        "episodes": 500,
        "gamma": 0.99,
        "epsilon_decay": 300,
        "solve_every": 10,
        "target_mode": "mc",
        "dqn_hidden": 128,
        "dqn_lr": 1e-3,
        "dqn_epsilon_decay": 200,
        "dqn_target_update": 10,
    },
    "LunarLander-v3": {
        "n_neurons": 8000,
        "episodes": 1000,
        "gamma": 0.99,
        "epsilon_decay": 300,
        "solve_every": 10,
        "target_mode": "mc",
        "dqn_hidden": 256,
        "dqn_lr": 5e-4,
        "dqn_epsilon_decay": 400,
        "dqn_target_update": 10,
    },
}


def run_env(
    env_name: str,
    seed: int = 0,
    skip_dqn: bool = False,
    n_episodes: int | None = None,
    n_neurons: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run NEF-FQI and DQN on a single environment."""
    cfg = ENV_CONFIGS.get(env_name, ENV_CONFIGS["CartPole-v1"])
    episodes = n_episodes or cfg["episodes"]
    neurons = n_neurons or cfg["n_neurons"]

    env = gym.make(env_name)
    d_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    results = {
        "env": env_name,
        "seed": seed,
        "n_neurons": neurons,
        "episodes": episodes,
    }

    # --- NEF-FQI ---
    if verbose:
        print(f"\n>>> NEF-FQI ({neurons} neurons) on {env_name}")
    nef_agent = NEFFQIAgent(
        d_obs=d_obs,
        n_actions=n_actions,
        n_neurons=neurons,
        gamma=cfg["gamma"],
        epsilon_decay=cfg["epsilon_decay"],
        solve_every=cfg["solve_every"],
        solve_batch=cfg.get("solve_batch"),
        tau=cfg.get("tau", 0.3),
        fqi_iters=cfg.get("fqi_iters", 1),
        target_mode=cfg.get("target_mode", "mc"),
        seed=seed,
    )
    nef_results = train_agent(nef_agent, env_name, n_episodes=episodes, seed=seed, verbose=verbose)
    results["nef_fqi"] = nef_results
    if verbose:
        print(
            f"  NEF-FQI final={nef_results['final_eval']:.1f}  "
            f"best={nef_results['best_eval']:.1f}  time={nef_results['time_s']:.1f}s"
        )

    # --- DQN ---
    if not skip_dqn:
        if verbose:
            print(f"\n>>> DQN ({cfg['dqn_hidden']}×2) on {env_name}")
        dqn_agent = DQNAgent(
            d_obs=d_obs,
            n_actions=n_actions,
            hidden=cfg["dqn_hidden"],
            gamma=cfg["gamma"],
            lr=cfg["dqn_lr"],
            epsilon_decay=cfg.get("dqn_epsilon_decay", cfg["epsilon_decay"]),
            target_update=cfg.get("dqn_target_update", 10),
            seed=seed,
        )
        dqn_results = train_agent(
            dqn_agent, env_name, n_episodes=episodes, seed=seed, verbose=verbose
        )
        results["dqn"] = dqn_results
        if verbose:
            print(
                f"  DQN final={dqn_results['final_eval']:.1f}  "
                f"best={dqn_results['best_eval']:.1f}  time={dqn_results['time_s']:.1f}s"
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="NEF-FQI RL benchmark")
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment name (default: run all)",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--n-neurons", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="results/rl/rl_results.json",
    )
    args = parser.parse_args()

    envs = [args.env] if args.env else list(ENV_CONFIGS.keys())
    all_results = []

    print("=" * 70)
    print("  NEF-FQI Reinforcement Learning Benchmark")
    print("=" * 70)

    for env_name in envs:
        result = run_env(
            env_name,
            seed=args.seed,
            skip_dqn=args.skip_dqn,
            n_episodes=args.episodes,
            n_neurons=args.n_neurons,
        )
        all_results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Environment':<20} {'Method':<12} {'Final':>8} {'Best':>8} {'Time':>8}")
    print("-" * 60)
    for r in all_results:
        nef = r["nef_fqi"]
        print(
            f"{r['env']:<20} {'NEF-FQI':<12} {nef['final_eval']:8.1f} "
            f"{nef['best_eval']:8.1f} {nef['time_s']:7.1f}s"
        )
        if "dqn" in r:
            dqn = r["dqn"]
            print(
                f"{'':<20} {'DQN':<12} {dqn['final_eval']:8.1f} "
                f"{dqn['best_eval']:8.1f} {dqn['time_s']:7.1f}s"
            )

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
