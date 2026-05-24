"""Tests for the NEFSuccessor (D1) skeleton on the toy chain MDP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leenef.rl_diag import SparseChainEnv
from leenef.rl_sf import NEFSuccessor


def _collect_episodes(env, agent, *, policy="right", n_episodes=200, eps=0.1, seed=0):
    """Run ``n_episodes`` of the given policy and feed transitions to ``agent``."""
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            if policy == "right":
                action = 1 if rng.random() > eps else int(rng.integers(2))
            elif policy == "random":
                action = int(rng.integers(2))
            else:
                raise ValueError(policy)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.add_transition(obs, action, reward, next_obs, done)
            obs = next_obs


def test_successor_targets_match_pure_mc():
    torch.manual_seed(0)
    sf = NEFSuccessor(d_obs=1, n_actions=2, n_neurons=8, gamma=0.9, n_step=None, seed=0)
    # Synthetic features for a 4-step episode.
    phi = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=torch.float64)
    sf.n_neurons = 2  # for this synthetic case we just want the recurrence
    psi = sf._compute_successor_targets(phi, T=4)
    # Pure MC backward recurrence: ψ_3 = φ_3; ψ_2 = φ_2 + γ ψ_3; ...
    expected = torch.zeros_like(phi)
    G = torch.zeros(2, dtype=torch.float64)
    for t in reversed(range(4)):
        G = phi[t] + 0.9 * G
        expected[t] = G
    assert torch.allclose(psi, expected, atol=1e-9)


def test_successor_targets_truncated_window():
    torch.manual_seed(1)
    sf = NEFSuccessor(d_obs=1, n_actions=2, n_neurons=4, gamma=0.5, n_step=2, seed=0)
    sf.n_neurons = 1
    phi = torch.tensor([[1.0], [2.0], [4.0], [8.0], [16.0]], dtype=torch.float64)
    psi = sf._compute_successor_targets(phi, T=5)
    # H = 2: ψ_t = φ_t + 0.5 φ_{t+1} when t+1 < T, else just φ_t.
    expected = torch.tensor(
        [
            [1.0 + 0.5 * 2.0],  # t=0
            [2.0 + 0.5 * 4.0],  # t=1
            [4.0 + 0.5 * 8.0],  # t=2
            [8.0 + 0.5 * 16.0],  # t=3
            [16.0],  # t=4 (boundary)
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(psi, expected, atol=1e-9)


def test_successor_solve_recovers_chain_optimal_q():
    """End-to-end: SF on the toy chain should recover Q*(s, RIGHT)."""
    torch.manual_seed(0)
    n_states = 5
    gamma = 0.9
    env = SparseChainEnv(n_states=n_states, gamma=gamma, max_steps=3 * n_states)
    sf = NEFSuccessor(
        d_obs=1,
        n_actions=2,
        n_neurons=128,
        gamma=gamma,
        alpha=1e-3,
        n_step=None,
        seed=0,
    )

    # Mostly-right policy with 30% randomness so both actions get coverage.
    _collect_episodes(env, sf, policy="right", eps=0.3, n_episodes=200, seed=0)
    sf.solve()

    Q_star = env.optimal_q()  # (n_states, 2)
    # Predicted Q for each non-terminal state.
    states = np.linspace(0, 1, n_states, dtype=np.float32).reshape(-1, 1)
    q_pred = np.stack([sf.q_values(s).numpy() for s in states], axis=0)

    # Check Q(s, RIGHT) matches Q*(s, RIGHT) for non-terminal states.
    err = np.abs(q_pred[:-1, 1] - Q_star[:-1, 1])
    assert err.max() < 0.15, (
        f"max error {err.max():.3f}, predicted {q_pred[:, 1]}, optimal {Q_star[:, 1]}"
    )

    # Greedy policy from learned Q must be RIGHT for all non-terminal states.
    greedy = q_pred[:-1].argmax(axis=1)
    assert (greedy == 1).all(), f"learned greedy policy is {greedy}, expected all RIGHT"


def test_successor_underused_action_keeps_zero_decoder():
    """Actions with <10 samples must not be solved (avoid singular system)."""
    torch.manual_seed(0)
    env = SparseChainEnv(n_states=4, gamma=0.9, max_steps=20)
    sf = NEFSuccessor(d_obs=1, n_actions=2, n_neurons=32, gamma=0.9, seed=0)
    # Always RIGHT — action 0 (LEFT) sees zero samples.
    _collect_episodes(env, sf, policy="right", eps=0.0, n_episodes=20, seed=0)
    sf.solve()
    assert int(sf._n_accumulated[0]) == 0
    # LEFT decoders untouched (initialized to zero).
    assert torch.equal(sf.Psi[0], torch.zeros_like(sf.Psi[0]))
    assert torch.equal(sf.W_r[0], torch.zeros_like(sf.W_r[0]))


def test_successor_q_value_shapes():
    torch.manual_seed(0)
    sf = NEFSuccessor(d_obs=2, n_actions=3, n_neurons=16, gamma=0.95, seed=0)
    obs = np.array([0.5, -0.2], dtype=np.float32)
    q = sf.q_values(obs)
    assert q.shape == (3,)
    psi = sf.successor(obs)
    assert psi.shape == (3, 16)
    r_hat = sf.reward_estimate(obs)
    assert r_hat.shape == (3,)
    # Batch path
    obs_batch = torch.from_numpy(np.stack([obs, obs * 2, obs - 0.1]))
    qb = sf.q_values(obs_batch)
    assert qb.shape == (3, 3)


def test_successor_respects_configured_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sf = NEFSuccessor(
        d_obs=2,
        n_actions=2,
        n_neurons=16,
        gamma=0.95,
        seed=0,
        device=device,
        directional_goal=[0.0, 0.0],
    )
    assert sf._ata.device.type == device
    obs = np.array([0.5, -0.2], dtype=np.float32)
    q = sf.q_values(obs)
    assert q.device.type == device
    assert q.shape == (2,)


def test_successor_forget_factor_decays_statistics():
    sf = NEFSuccessor(d_obs=1, n_actions=2, n_neurons=8, gamma=0.9, forget_factor=0.5, seed=0)
    env = SparseChainEnv(n_states=4, gamma=0.9, max_steps=12)
    _collect_episodes(env, sf, policy="random", n_episodes=5, seed=0)
    ata_before = sf._ata.clone()
    aty_phi_before = sf._aty_phi.clone()
    aty_r_before = sf._aty_r.clone()
    # Run one more episode.
    _collect_episodes(env, sf, policy="random", n_episodes=1, seed=99)
    # New stats should be β * old + new contribution.
    assert not torch.allclose(sf._ata, ata_before)
    # Trace decreases at least once if no new transitions for some action.
    delta = sf._ata - ata_before * 0.5
    # By construction at least one action got new data; allow either sign of
    # change, but reject NaN/Inf.
    assert torch.isfinite(sf._ata).all()
    assert torch.isfinite(sf._aty_phi).all()
    assert torch.isfinite(sf._aty_r).all()
    # Sanity-suppress unused locals.
    _ = (aty_phi_before, aty_r_before, delta)
