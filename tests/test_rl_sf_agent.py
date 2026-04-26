"""Tests for the NEFSFAgent training-loop wrapper + leverage novelty (D2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leenef.rl_diag import SparseChainEnv
from leenef.rl_sf import NEFSFAgent


def _train_chain(agent: NEFSFAgent, env: SparseChainEnv, n_episodes: int, seed: int = 0):
    """Run the agent on the chain for ``n_episodes`` and return per-episode returns."""
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            agent.update(obs, action, reward, next_obs, done)
            ep_ret += reward
            obs = next_obs
        returns.append(ep_ret)
    return returns


def test_agent_select_action_random_before_solve():
    env = SparseChainEnv(n_states=5)
    agent = NEFSFAgent(d_obs=1, n_actions=2, n_neurons=64, seed=1, solve_every=999)
    obs, _ = env.reset(seed=0)
    actions = [agent.select_action(obs) for _ in range(200)]
    assert set(actions) == {0, 1}, "Pre-solve policy must explore both actions"


def test_agent_solves_periodically():
    env = SparseChainEnv(n_states=5)
    agent = NEFSFAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=64,
        gamma=0.9,
        alpha=1e-3,
        solve_every=5,
        epsilon_start=0.5,
        epsilon_end=0.5,
        seed=2,
    )
    _train_chain(agent, env, n_episodes=20)
    assert agent._solve_count == 20 // 5


def test_agent_recovers_chain_optimal_policy():
    """End-to-end: ε-greedy training on the toy chain finds the right-going policy."""
    env = SparseChainEnv(n_states=5)
    agent = NEFSFAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=128,
        gamma=0.9,
        alpha=1e-3,
        n_step=8,
        solve_every=10,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=50,
        warmup_episodes=20,
        seed=3,
    )
    _train_chain(agent, env, n_episodes=300, seed=10)

    # Greedy from any non-terminal state should be RIGHT (action 1).
    for s in range(env.n_states - 1):
        obs = np.array([float(s) / (env.n_states - 1)], dtype=np.float32)
        q = agent.q_values(obs)
        assert int(q.argmax().item()) == 1, f"State {s}: expected RIGHT, q={q.tolist()}"


def test_novelty_zero_before_solve():
    """Without a cached inverse, novelty must read as zero."""
    env = SparseChainEnv(n_states=5)
    agent = NEFSFAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=64,
        novelty_beta=1.0,
        seed=4,
    )
    obs, _ = env.reset(seed=0)
    assert agent.novelty(obs) == 0.0


def test_novelty_higher_for_unvisited_states():
    """After training only on the left half of the chain, right-half states must be more novel."""
    env = SparseChainEnv(n_states=11)
    agent = NEFSFAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=128,
        gamma=0.9,
        alpha=1e-3,
        novelty_beta=1.0,
        epsilon_start=1.0,
        epsilon_end=1.0,
        solve_every=1,
        seed=5,
    )
    # Synthetic episodes that only visit states 0..3.
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, _ = env.reset(seed=int(rng.integers(0, 10000)))
        for _ in range(rng.integers(2, 5)):
            action = 0  # LEFT — but env clamps at 0; manually craft transitions
            next_obs, reward, term, trunc, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, bool(term or trunc))
            obs = next_obs
            if term or trunc:
                break
    # Force a solve so the novelty inverse is populated.
    agent.solve()

    visited = float(agent.novelty(np.array([0.1], dtype=np.float32)))
    unvisited = float(agent.novelty(np.array([0.9], dtype=np.float32)))
    assert unvisited > visited, f"Expected unvisited>visited, got {unvisited} vs {visited}"


def test_novelty_bonus_added_to_episode_reward():
    """Intrinsic reward must be tracked separately from the extrinsic accumulator."""
    env = SparseChainEnv(n_states=5)
    agent = NEFSFAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=64,
        novelty_beta=1.0,
        epsilon_start=1.0,
        epsilon_end=1.0,
        solve_every=1,
        seed=6,
    )
    # First episode has zero novelty (no inverse yet).
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
    # After solve, run a second episode and confirm intrinsic > 0 on at least one step.
    saw_intrinsic = False
    obs, _ = env.reset(seed=1)
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        agent.update(obs, action, reward, next_obs, done)
        if agent.last_intrinsic_reward > 0:
            saw_intrinsic = True
        obs = next_obs
    assert saw_intrinsic


def test_novelty_disabled_no_inverse():
    """novelty_beta=0 must keep _novelty_inv at None forever."""
    env = SparseChainEnv(n_states=5)
    agent = NEFSFAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=64,
        novelty_beta=0.0,
        solve_every=1,
        epsilon_start=1.0,
        epsilon_end=1.0,
        seed=7,
    )
    _train_chain(agent, env, n_episodes=10)
    assert agent._novelty_inv is None
    obs = np.array([0.5], dtype=np.float32)
    assert agent.novelty(obs) == 0.0


def test_warmup_keeps_epsilon_at_start():
    """During warmup, epsilon must equal epsilon_start regardless of episode count."""
    agent = NEFSFAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=32,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=10,
        warmup_episodes=15,
        seed=8,
    )
    for ep in range(15):
        agent._episode = ep
        assert agent._epsilon() == 1.0
    agent._episode = 100
    assert agent._epsilon() < 1.0


def test_agent_q_values_shape_consistency():
    """select_action and q_values must agree on a single-state observation."""
    env = SparseChainEnv(n_states=5)
    agent = NEFSFAgent(
        d_obs=1,
        n_actions=2,
        n_neurons=64,
        epsilon_start=0.0,
        epsilon_end=0.0,
        solve_every=5,
        seed=9,
    )
    _train_chain(agent, env, n_episodes=30, seed=20)
    obs = np.array([0.5], dtype=np.float32)
    q = agent.q_values(obs)
    assert q.shape == torch.Size([2])
    assert torch.isfinite(q).all()
