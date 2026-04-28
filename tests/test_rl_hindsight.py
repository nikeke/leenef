"""Tests for :class:`NEFHindsightSFAgent` (D4 hindsight retargeting)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from leenef import NEFHindsightSFAgent
from leenef.rl_diag import SparseChainEnv


def _common_kwargs(**overrides):
    kw = dict(
        d_obs=4,
        n_actions=2,
        n_neurons=128,
        gamma=0.95,
        alpha=1e-2,
        n_step=10,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=20,
        solve_every=1,
        warmup_episodes=0,
        seed=0,
    )
    kw.update(overrides)
    return kw


def test_init_validates_buffer_size():
    with pytest.raises(ValueError):
        NEFHindsightSFAgent(goal_buffer_size=0, **_common_kwargs())


def test_init_validates_horizon():
    with pytest.raises(ValueError):
        NEFHindsightSFAgent(goal_horizon=0, **_common_kwargs())


def test_init_validates_warmup_solves():
    with pytest.raises(ValueError):
        NEFHindsightSFAgent(goal_warmup_solves=-1, **_common_kwargs())


def test_falls_back_to_extrinsic_q_before_first_solve():
    agent = NEFHindsightSFAgent(**_common_kwargs(goal_warmup_solves=1))
    obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    a = agent.select_action(obs)
    assert 0 <= a < 2
    diag = agent.goal_diagnostics()
    assert diag["goal_buffer_size"] >= 1
    assert diag["current_goal_active"] is False


def test_goal_q_values_shape_after_solve():
    agent = NEFHindsightSFAgent(**_common_kwargs())
    rng = np.random.default_rng(0)
    for _ in range(40):
        obs = rng.normal(size=4).astype(np.float32)
        next_obs = rng.normal(size=4).astype(np.float32)
        agent.update(obs, int(rng.integers(2)), float(rng.normal()), next_obs, False)
    agent.update(obs, 0, 0.0, next_obs, True)
    agent.solve()
    agent._resample_goal()
    if agent._current_goal_phi is None:
        pytest.skip("buffer empty; goal sampling skipped")
    gq = agent.goal_q_values(obs)
    assert gq is not None
    assert gq.shape == (2,)


def test_goal_resamples_after_horizon():
    agent = NEFHindsightSFAgent(
        **_common_kwargs(goal_horizon=3, goal_warmup_solves=0, epsilon_start=0.0, epsilon_end=0.0)
    )
    # Force a non-zero solve so goal mechanism activates.
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(size=4).astype(np.float32)
        agent.update(obs, 0, 1.0, obs, False)
    agent.update(obs, 0, 1.0, obs, True)
    agent.solve()

    initial_resample_count = agent._goal_resample_count
    obs = rng.normal(size=4).astype(np.float32)
    for _ in range(10):
        agent.select_action(obs)
    # After 10 steps with horizon=3, should have resampled at least 3 times.
    assert agent._goal_resample_count - initial_resample_count >= 3


def test_extrinsic_q_unchanged_by_goal_mechanism():
    """Hindsight only affects behavior; extrinsic Q (= ψ·w_a) is solved as usual."""
    agent_h = NEFHindsightSFAgent(**_common_kwargs())
    rng = np.random.default_rng(42)
    obs_seq = [rng.normal(size=4).astype(np.float32) for _ in range(50)]
    for i, obs in enumerate(obs_seq[:-1]):
        a = int(rng.integers(2))
        r = float(rng.normal())
        done = i == 48
        agent_h.update(obs, a, r, obs_seq[i + 1], done)
    agent_h.solve()

    # Extrinsic Q should match ψ·w_a (the parent class formula).
    s = obs_seq[0]
    phi = agent_h.features.encode(torch.from_numpy(s).float())
    psi = torch.einsum("n,anm->am", phi, agent_h.Psi)
    expected = (psi * agent_h.W_r).sum(dim=1)
    actual = agent_h.q_values(s)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_goal_action_uses_goal_q_when_greedy():
    """When ε=0 and goal is active, action = argmax goal_q_values."""
    agent = NEFHindsightSFAgent(
        **_common_kwargs(
            goal_horizon=100, goal_warmup_solves=0, epsilon_start=0.0, epsilon_end=0.0
        )
    )
    rng = np.random.default_rng(0)
    obs_seq = [rng.normal(size=4).astype(np.float32) for _ in range(30)]
    for i, obs in enumerate(obs_seq[:-1]):
        agent.update(obs, int(rng.integers(2)), 1.0, obs_seq[i + 1], i == 28)
    agent.solve()
    agent._resample_goal()
    if agent._current_goal_phi is None:
        pytest.skip("no goal active")
    gq = agent.goal_q_values(obs_seq[0])
    expected = int(gq.argmax().item())
    actual = agent.select_action(obs_seq[0])
    assert actual == expected


def test_init_validates_goal_mode():
    with pytest.raises(ValueError):
        NEFHindsightSFAgent(goal_mode="invalid", **_common_kwargs())


def test_init_validates_goal_random_prob():
    with pytest.raises(ValueError):
        NEFHindsightSFAgent(goal_random_prob=1.5, **_common_kwargs())


def test_random_direction_mode_unit_norm_no_buffer():
    """random_direction mode draws a unit goal vector even with empty buffer."""
    agent = NEFHindsightSFAgent(
        **_common_kwargs(goal_warmup_solves=0, goal_mode="random_direction")
    )
    # Force one solve so the goal mechanism activates.
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(size=4).astype(np.float32)
        agent.update(obs, 0, 1.0, obs, False)
    agent.update(obs, 0, 1.0, obs, True)
    agent.solve()

    agent._resample_goal()
    assert agent._current_goal_phi is not None
    norm = float(torch.linalg.vector_norm(agent._current_goal_phi).item())
    assert abs(norm - 1.0) < 1e-9
    assert agent._goal_random_count >= 1


def test_mixed_mode_uses_both_branches():
    """In mixed mode, with prob=0.5 over many resamples, both counters grow."""
    agent = NEFHindsightSFAgent(
        **_common_kwargs(
            goal_horizon=1,
            goal_warmup_solves=0,
            goal_mode="mixed",
            goal_random_prob=0.5,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
    )
    rng = np.random.default_rng(0)
    for _ in range(30):
        obs = rng.normal(size=4).astype(np.float32)
        agent.update(obs, 0, 1.0, obs, False)
    agent.update(obs, 0, 1.0, obs, True)
    agent.solve()

    for _ in range(80):
        agent.select_action(rng.normal(size=4).astype(np.float32))
    diag = agent.goal_diagnostics()
    assert diag["goal_random_count"] >= 5
    assert diag["goal_buffer_count"] >= 5


def test_init_validates_goal_action_mode():
    with pytest.raises(ValueError):
        NEFHindsightSFAgent(goal_action_mode="invalid", **_common_kwargs())


def test_commit_mode_takes_goal_action_every_step():
    """In commit mode + after warmup, every action is goal-Q greedy (modulo ties)."""
    agent = NEFHindsightSFAgent(
        **_common_kwargs(
            goal_horizon=1000,
            goal_warmup_solves=0,
            goal_mode="random_direction",
            goal_action_mode="commit",
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
    )
    rng = np.random.default_rng(1)
    for _ in range(20):
        obs = rng.normal(size=4).astype(np.float32)
        agent.update(obs, 0, 1.0, obs, False)
    agent.update(obs, 0, 1.0, obs, True)
    agent.solve()

    obs = rng.normal(size=4).astype(np.float32)
    a = agent.select_action(obs)
    gq = agent.goal_q_values(obs)
    assert gq is not None
    assert a == int(gq.argmax().item())
    diag = agent.goal_diagnostics()
    assert diag["goal_action_mode"] == "commit"
    assert diag["goal_action_count"] >= 1


def test_commit_mode_holds_goal_for_horizon():
    """In commit mode, ε is ignored and goal holds for goal_horizon steps."""
    horizon = 5
    agent = NEFHindsightSFAgent(
        **_common_kwargs(
            goal_horizon=horizon,
            goal_warmup_solves=0,
            goal_mode="random_direction",
            goal_action_mode="commit",
            epsilon_start=1.0,
            epsilon_end=1.0,
        )
    )
    rng = np.random.default_rng(2)
    for _ in range(15):
        obs = rng.normal(size=4).astype(np.float32)
        agent.update(obs, 0, 1.0, obs, False)
    agent.update(obs, 0, 1.0, obs, True)
    agent.solve()

    n_steps = 30
    for _ in range(n_steps):
        agent.select_action(rng.normal(size=4).astype(np.float32))
    diag = agent.goal_diagnostics()
    # Should resample roughly n_steps / horizon times, not every step.
    expected = n_steps // horizon
    assert diag["goal_resample_count"] <= expected + 2
    assert diag["goal_resample_count"] >= max(1, expected - 1)


def test_chain_recovers_optimal_policy():
    """Sanity: hindsight agent's underlying extrinsic Q still solves the toy chain.

    The hindsight wrapper biases ε-greedy *behavior* toward goals sampled
    from visited states, which on a dense-reward chain can pull the agent
    backward.  The invariant we care about is that the inherited
    NEFSFAgent solve is undisturbed: greedy *extrinsic* Q must still
    point RIGHT.
    """
    env = SparseChainEnv(n_states=5)
    agent = NEFHindsightSFAgent(
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
        goal_buffer_size=128,
        goal_horizon=5,
        goal_warmup_solves=1,
        seed=3,
    )
    rng = np.random.default_rng(10)
    for _ in range(300):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            a = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            agent.update(obs, a, reward, next_obs, done)
            obs = next_obs

    # Inherited extrinsic Q (NOT goal-conditioned) must point RIGHT on
    # every non-terminal state.
    for s in range(env.n_states - 1):
        obs = np.array([float(s) / (env.n_states - 1)], dtype=np.float32)
        q = agent.q_values(obs)
        assert int(q.argmax().item()) == 1, f"State {s}: expected RIGHT, q={q.tolist()}"
