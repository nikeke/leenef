"""Sparse-reward diagnostics and a deterministic toy chain MDP.

Two responsibilities:

1. **Failure-mode diagnostics.**  A lightweight collector that wraps a
   training loop and answers the question "*which* failure mode is
   binding on this environment?" — see :class:`SparseRewardDiagnostics`.
   The three modes we want to distinguish are:

   - *No informative reward events are ever discovered.*  Exploration
     never reaches any positive reward; ``AᵀY`` stays at the noise
     floor and the analytical solve has nothing to fit.  Fix axis is
     exploration / novelty.
   - *Reward events occur but credit does not propagate far enough.*
     Goal is reached occasionally, but ``Y`` is non-zero only on a
     handful of late states and the fit cannot project it backward in
     time.  Fix axis is target structure (successor features,
     trajectory pooling, hindsight).
   - *Credit exists but features cannot separate good from bad
     regions.*  Both ``AᵀA`` and a non-trivial ``AᵀY`` are present,
     yet ``Q(s, a)`` from the solve does not improve the policy.  Fix
     axis is representation (data-driven biases, multi-head Q).

2. **Sparse-reward unit environment.**  :class:`SparseChainEnv` is a
   k-state deterministic chain with a single +1 terminal reward and
   zero per-step reward.  It exposes a closed-form optimal Q via
   value iteration so any new mechanism (FQI, NEF-SF, novelty bonus,
   …) can be unit-tested before being run on MountainCar/Acrobot.

Both pieces are intentionally lightweight: no PyTorch state, no
gymnasium dependency on the diagnostics path, no benchmark runner.
They are meant to be composed *into* existing harnesses rather than
replace them.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Sparse-reward chain MDP
# ---------------------------------------------------------------------------


class SparseChainEnv:
    """Deterministic k-state chain with a single sparse terminal reward.

    States ``{0, 1, …, n_states-1}``.  Two discrete actions:

    - action ``0`` (LEFT): ``s → max(0, s-1)``
    - action ``1`` (RIGHT): ``s → min(n_states-1, s+1)``

    Reward is ``+1`` on transitioning to the terminal state
    ``n_states-1`` and ``0`` everywhere else.  The episode terminates
    when the terminal state is reached, or when ``max_steps`` is hit
    (truncation).

    The observation is a 1-D float32 array ``[s / (n_states - 1)]``
    in ``[0, 1]`` so it composes with NEF observation normalization
    without special-casing.

    The environment exposes a closed-form optimal Q-table via
    :meth:`optimal_q` (value iteration to convergence; cheap because
    ``n_states`` is small and dynamics are deterministic).  This makes
    it a precise unit env for the analytical solve: fitted Q from any
    of our methods should match :meth:`optimal_q` within regularization
    tolerance once enough successful trajectories are collected.

    Mirrors the ``gymnasium`` reset/step contract closely enough that
    it can be passed to the existing rl benchmarks: returns
    ``(obs, info)`` from :meth:`reset` and
    ``(obs, reward, terminated, truncated, info)`` from :meth:`step`.
    """

    def __init__(
        self,
        n_states: int = 10,
        max_steps: int | None = None,
        gamma: float = 0.99,
        seed: int | None = None,
    ):
        if n_states < 3:
            raise ValueError(f"n_states must be >= 3, got {n_states}")
        self.n_states = int(n_states)
        self.max_steps = int(max_steps) if max_steps is not None else 4 * n_states
        self.gamma = float(gamma)
        self.observation_space = _BoxLike(low=0.0, high=1.0, shape=(1,))
        self.action_space = _DiscreteLike(n=2, seed=seed)
        self._state = 0
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

    def _obs(self) -> np.ndarray:
        return np.array([self._state / (self.n_states - 1)], dtype=np.float32)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        self._state = 0
        self._step_count = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if action == 0:
            self._state = max(0, self._state - 1)
        elif action == 1:
            self._state = min(self.n_states - 1, self._state + 1)
        else:
            raise ValueError(f"action must be 0 or 1, got {action!r}")

        self._step_count += 1
        terminated = self._state == self.n_states - 1
        reward = 1.0 if terminated else 0.0
        truncated = (not terminated) and self._step_count >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def close(self) -> None:  # for gym-API compatibility
        return None

    # -- closed-form optimal Q ------------------------------------------------

    def optimal_q(
        self,
        gamma: float | None = None,
        tol: float = 1e-12,
        max_iter: int = 10_000,
    ) -> np.ndarray:
        """Optimal Q-table ``(n_states, 2)`` via value iteration.

        Terminal state has Q = 0 (no further reward).  Returns a NumPy
        array of shape ``(n_states, 2)``.
        """
        gamma = self.gamma if gamma is None else float(gamma)
        n = self.n_states
        Q = np.zeros((n, 2), dtype=np.float64)
        for _ in range(max_iter):
            Q_new = np.zeros_like(Q)
            for s in range(n - 1):  # terminal stays at 0
                for a in (0, 1):
                    s_next = max(0, s - 1) if a == 0 else min(n - 1, s + 1)
                    r = 1.0 if s_next == n - 1 else 0.0
                    if s_next == n - 1:
                        Q_new[s, a] = r  # terminal: no bootstrap
                    else:
                        Q_new[s, a] = r + gamma * Q[s_next].max()
            if np.max(np.abs(Q_new - Q)) < tol:
                Q = Q_new
                break
            Q = Q_new
        return Q

    def optimal_policy(self) -> np.ndarray:
        """Greedy policy w.r.t. :meth:`optimal_q` as a length-``n_states`` array."""
        return self.optimal_q().argmax(axis=1)


@dataclass
class _BoxLike:
    low: float
    high: float
    shape: tuple

    @property
    def dtype(self) -> np.dtype:
        return np.dtype("float32")


class _DiscreteLike:
    """Minimal stand-in for ``gym.spaces.Discrete`` used by RL helpers."""

    def __init__(self, n: int, seed: int | None = None):
        self.n = int(n)
        self._rng = np.random.default_rng(seed)

    def seed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> int:
        return int(self._rng.integers(self.n))


# ---------------------------------------------------------------------------
# Diagnostics collector
# ---------------------------------------------------------------------------


@dataclass
class SparseRewardDiagnostics:
    """Per-run failure-mode diagnostics for sparse-reward training.

    Wraps a training loop with a small amount of bookkeeping.  Has no
    side effects on the agent; just records.  Pair with
    :func:`feature_coverage_stats` and :func:`leverage_stats` (which
    take an agent snapshot) at solve points or end of training.

    Usage::

        diag = SparseRewardDiagnostics()
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                diag.record_step(reward, action)
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
            diag.end_episode(terminated=term)
            agent.end_episode()
        summary = diag.summary()
    """

    # Per-step rolling state
    _ep_reward: float = 0.0
    _ep_length: int = 0
    _ep_positive_events: int = 0
    _ep_action_counts: Counter = field(default_factory=Counter)

    # Per-episode history (parallel lists; kept compact for long runs)
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_positive_events: list[int] = field(default_factory=list)
    episode_terminated: list[bool] = field(default_factory=list)

    # Cross-episode aggregates
    total_positive_events: int = 0
    first_positive_episode: int | None = None
    action_counts: Counter = field(default_factory=Counter)

    def record_step(self, reward: float, action: int | None = None) -> None:
        """Record a single environment transition."""
        self._ep_reward += float(reward)
        self._ep_length += 1
        if reward > 0:
            self._ep_positive_events += 1
            self.total_positive_events += 1
        if action is not None:
            self._ep_action_counts[int(action)] += 1
            self.action_counts[int(action)] += 1

    def end_episode(self, *, terminated: bool = False) -> None:
        """Close the current episode and roll its stats into history."""
        ep_idx = len(self.episode_rewards)
        self.episode_rewards.append(self._ep_reward)
        self.episode_lengths.append(self._ep_length)
        self.episode_positive_events.append(self._ep_positive_events)
        self.episode_terminated.append(bool(terminated))
        if self._ep_positive_events > 0 and self.first_positive_episode is None:
            self.first_positive_episode = ep_idx
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_positive_events = 0
        self._ep_action_counts = Counter()

    # -- queries --------------------------------------------------------------

    def success_rate(self, last_k: int = 100, threshold: float = 0.0) -> float:
        """Fraction of the last ``last_k`` episodes whose return > threshold.

        For sparse-binary tasks (toy chain, MountainCar with reach-goal
        bonus) the default ``threshold=0.0`` matches "did the agent
        ever see the goal in this episode."  For dense classic-control
        tasks, callers can pass a meaningful threshold.
        """
        if not self.episode_rewards:
            return 0.0
        window = self.episode_rewards[-last_k:]
        return float(np.mean([r > threshold for r in window]))

    def positive_event_rate(self, last_k: int = 100) -> float:
        """Average per-episode positive-event count over the last ``last_k`` episodes."""
        if not self.episode_positive_events:
            return 0.0
        window = self.episode_positive_events[-last_k:]
        return float(np.mean(window))

    def action_distribution(self, last_k: int | None = None) -> dict[int, float]:
        """Empirical action distribution (over all time, or last K episodes)."""
        if last_k is None:
            counts = self.action_counts
        else:
            # Recompute over the last K episodes' worth of action records.
            # Cheap because we keep total counts; drop the truncation case.
            counts = self.action_counts
        total = sum(counts.values())
        if total == 0:
            return {}
        return {int(a): c / total for a, c in counts.items()}

    def summary(self, last_k: int = 100) -> dict:
        """Compact dict suitable for JSON logging."""
        return {
            "n_episodes": len(self.episode_rewards),
            "total_positive_events": self.total_positive_events,
            "first_positive_episode": self.first_positive_episode,
            "success_rate_last_k": round(self.success_rate(last_k), 4),
            "positive_event_rate_last_k": round(self.positive_event_rate(last_k), 4),
            "mean_episode_length_last_k": round(
                float(np.mean(self.episode_lengths[-last_k:])) if self.episode_lengths else 0.0,
                3,
            ),
            "termination_rate_last_k": round(
                float(np.mean(self.episode_terminated[-last_k:]))
                if self.episode_terminated
                else 0.0,
                4,
            ),
            "action_distribution": {a: round(p, 4) for a, p in self.action_distribution().items()},
        }


# ---------------------------------------------------------------------------
# Feature / solve diagnostics (stateless; act on agent snapshots)
# ---------------------------------------------------------------------------


def leverage_stats(
    ata: Tensor,
    alpha: float,
    phi_samples: Tensor,
) -> dict[str, float]:
    """Statistics of the leverage score ``φᵀ (AᵀA + αI)⁻¹ φ``.

    The leverage of a sample ``φ`` in the current sufficient-statistic
    regime is exactly the variance of the analytical estimator at that
    point — large leverage means the sample is far from previously
    visited regions.  This is also the D2 novelty score (up to the
    inverse-square-root).

    Args:
        ata: ``(n_neurons, n_neurons)`` Gram matrix.
        alpha: Tikhonov regularization (will be added trace-scaled
            following the same rule used in :class:`NEFFQIAgent`).
        phi_samples: ``(N, n_neurons)`` features to score.

    Returns:
        Dict of mean / std / min / max / percentile leverage.  All
        quantities are ordinary floats for JSON serialization.
    """
    n_neurons = ata.shape[0]
    M = ata.clone().double()
    reg = alpha * torch.trace(M) / n_neurons
    M.diagonal().add_(reg.clamp(min=alpha))
    phi64 = phi_samples.double()
    # Solve M x = φ, then leverage = φ · x.
    sol = torch.linalg.solve(M, phi64.T)  # (n_neurons, N)
    leverage = (phi64 * sol.T).sum(dim=1).clamp(min=0.0)
    novelty = 1.0 / (leverage.sqrt() + 1e-12)
    return {
        "leverage_mean": round(leverage.mean().item(), 6),
        "leverage_std": round(leverage.std().item(), 6),
        "leverage_min": round(leverage.min().item(), 6),
        "leverage_max": round(leverage.max().item(), 6),
        "leverage_p50": round(torch.quantile(leverage, 0.5).item(), 6),
        "leverage_p95": round(torch.quantile(leverage, 0.95).item(), 6),
        "novelty_mean": round(novelty.mean().item(), 6),
        "novelty_p95": round(torch.quantile(novelty, 0.95).item(), 6),
    }


def solve_conditioning(ata: Tensor, alpha: float) -> dict[str, float]:
    """Conditioning of the regularized SPD solve actually performed.

    Reports the trace, the minimum eigenvalue (post-regularization),
    and the resulting condition number.  Mirrors the trace-scaled
    regularization rule used in :class:`NEFFQIAgent`.
    """
    n_neurons = ata.shape[0]
    M = ata.clone().double()
    tr = torch.trace(M).item()
    reg = alpha * tr / n_neurons
    eff_reg = max(reg, alpha)
    M.diagonal().add_(eff_reg)
    eigs = torch.linalg.eigvalsh(M)
    eig_min = float(eigs.min().item())
    eig_max = float(eigs.max().item())
    cond = eig_max / max(eig_min, 1e-30)
    return {
        "trace": round(tr, 4),
        "effective_alpha": round(eff_reg, 6),
        "eig_min": round(eig_min, 6),
        "eig_max": round(eig_max, 4),
        "cond_number": round(cond, 2),
    }


def target_variance_by_distance(
    transitions: list[tuple[np.ndarray, int, float]],
    *,
    gamma: float,
    n_step: int | None = None,
    n_bins: int = 5,
) -> dict[str, list]:
    """Variance of n-step / MC targets bucketed by distance-to-reward.

    Used to diagnose failure mode 2 ("credit doesn't propagate"):
    if early-trajectory targets are nearly constant while late-trajectory
    targets carry the reward, the analytical solve will only fit late
    states.

    Args:
        transitions: list of ``(obs, action, reward)`` triples for a
            *single* episode.
        gamma: discount factor for return computation.
        n_step: if set, use truncated n-step returns.  Otherwise MC.
        n_bins: number of distance-from-reward bins.

    Returns:
        Dict with ``bin_edges``, per-bin mean/std/count of targets,
        and the binned distance index per transition.
    """
    T = len(transitions)
    if T == 0:
        return {"bin_edges": [], "mean": [], "std": [], "count": []}
    rewards = [t[2] for t in transitions]

    targets = np.zeros(T, dtype=np.float64)
    if n_step is None:
        G = 0.0
        for i in reversed(range(T)):
            G = rewards[i] + gamma * G
            targets[i] = G
    else:
        for t in range(T):
            G = 0.0
            horizon = min(n_step, T - t)
            for k in reversed(range(horizon)):
                G = rewards[t + k] + gamma * G
            targets[t] = G

    distances = np.arange(T - 1, -1, -1, dtype=np.int64)  # 0 = closest to terminal
    edges = np.linspace(0, max(T - 1, 1), n_bins + 1)
    bin_idx = np.clip(np.digitize(distances, edges) - 1, 0, n_bins - 1)
    means = []
    stds = []
    counts = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            means.append(round(float(targets[mask].mean()), 6))
            stds.append(round(float(targets[mask].std()), 6))
            counts.append(int(mask.sum()))
        else:
            means.append(0.0)
            stds.append(0.0)
            counts.append(0)
    return {
        "bin_edges": [round(float(e), 3) for e in edges],
        "mean": means,
        "std": stds,
        "count": counts,
    }


def per_action_sample_counts(
    n_accumulated: Tensor | None,
    buffer_actions: list[int] | None = None,
    n_actions: int | None = None,
) -> dict[int, int]:
    """How many transitions each action contributes to the solve.

    Detects degenerate action distributions early.  Pass either the
    agent's ``_n_accumulated`` (RLS path) or a list of buffered
    actions; the first found wins.
    """
    if n_accumulated is not None:
        return {int(a): int(n_accumulated[a].item()) for a in range(len(n_accumulated))}
    if buffer_actions is not None:
        ctr = Counter(buffer_actions)
        if n_actions is None:
            n_actions = max(ctr.keys()) + 1 if ctr else 0
        return {a: ctr.get(a, 0) for a in range(n_actions)}
    return {}


def feature_coverage_stats(
    ata: Tensor,
    *,
    activity: Tensor | None = None,
    threshold: float = 1e-6,
) -> dict[str, float]:
    """Quick coverage summary of the encoded feature space.

    Args:
        ata: ``(n_neurons, n_neurons)`` Gram matrix (per-action or summed).
        activity: optional per-neuron activity vector.
        threshold: relative-to-max threshold under which a neuron is
            considered "silent."
    """
    diag = ata.diagonal().double()
    silent_mask = diag < threshold * diag.max().clamp(min=1e-30)
    out = {
        "trace_per_neuron": round(float(diag.mean().item()), 6),
        "diag_p5": round(float(torch.quantile(diag, 0.05).item()), 6),
        "diag_p95": round(float(torch.quantile(diag, 0.95).item()), 6),
        "n_silent_neurons": int(silent_mask.sum().item()),
        "frac_silent_neurons": round(float(silent_mask.float().mean().item()), 4),
    }
    if activity is not None:
        a = activity.double()
        out["activity_mean"] = round(float(a.mean().item()), 6)
        out["activity_p5"] = round(float(torch.quantile(a, 0.05).item()), 6)
        out["frac_inactive"] = round(
            float((a < threshold * a.max().clamp(min=1e-30)).float().mean().item()), 4
        )
    return out


# ---------------------------------------------------------------------------
# Sparse-reward primary metrics (for benchmark output)
# ---------------------------------------------------------------------------


def success_rate_from_eval_history(
    eval_history: list[dict],
    *,
    threshold: float = 0.0,
    last_k: int = 3,
    key: str = "eval",
) -> float | None:
    """Mean success indicator over the last ``last_k`` eval checkpoints.

    For per-eval logs that record raw returns under ``key``, returns
    the fraction (0–1) of those checkpoints with ``return > threshold``.
    Mirrors :func:`benchmarks.rl_repro.robust_eval_from_history` shape
    so it slots into existing summary code.
    """
    values = [float(e[key]) for e in eval_history[-last_k:] if e.get(key) is not None]
    if not values:
        return None
    return round(float(np.mean([v > threshold for v in values])), 4)


def first_positive_eval_episode(
    eval_history: list[dict],
    *,
    threshold: float = 0.0,
    key: str = "eval",
    episode_key: str = "episode",
) -> int | None:
    """Episode index of the first eval checkpoint that crossed ``threshold``."""
    for entry in eval_history:
        v = entry.get(key)
        if v is not None and float(v) > threshold:
            return int(entry.get(episode_key, -1))
    return None


def sparse_metric_summary(
    eval_history: list[dict],
    *,
    threshold: float = 0.0,
    last_k: int = 3,
) -> dict[str, float | int | None]:
    """Combined sparse-reward summary for a single run.

    Returns ``success_rate`` (primary), ``first_positive_episode``
    (leading indicator), and ``robust_eval`` (mean of last K) so a
    sparse run can be ranked without falling back to misleading mean
    return on a constant-floor task.
    """
    return {
        "success_rate": success_rate_from_eval_history(
            eval_history, threshold=threshold, last_k=last_k
        ),
        "first_positive_episode": first_positive_eval_episode(eval_history, threshold=threshold),
        "robust_eval": (
            round(
                float(
                    np.mean(
                        [
                            float(e["eval"])
                            for e in eval_history[-last_k:]
                            if e.get("eval") is not None
                        ]
                    )
                ),
                4,
            )
            if eval_history
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Compatibility deque for trajectory windows (kept here to avoid bloat
# in rl_sf when it imports from this module).
# ---------------------------------------------------------------------------


def rolling_window(seq: list, k: int) -> deque:
    """Return a length-``k`` deque seeded with the last elements of ``seq``."""
    return deque(seq[-k:], maxlen=k)
