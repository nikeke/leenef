"""NEF-based reinforcement learning components.

Provides :class:`NEFFeatures` (fixed random feature encoder with online
state normalization) and :class:`NEFFQIAgent` (Fitted Q-Iteration with
analytical least-squares solve).

The core idea: replace DQN's gradient-based Q-network updates with NEF's
analytical solve.  Both approaches approximate Q(s, a); the difference
is *how* the approximation weights are updated — SGD mini-batches vs.
one-shot least-squares on the full replay buffer.

Two target modes are supported:

- **mc** (Monte Carlo): regression targets are undiscounted episode
  returns G_t = Σ γ^k r_{t+k}.  No bootstrapping, no target network.
- **td** (Temporal Difference): bootstrapped targets y = r + γ max Q(s').
  Requires a Polyak-averaged target weight matrix for stability.

Two optional mechanisms improve performance:

- **Exponential forgetting** (RLS): sufficient statistics decay as
  AᵀA ← β·AᵀA + batch, eliminating the replay buffer while adapting
  to non-stationary targets.
- **Dead neuron recentering**: neurons with low mean activation are
  recentered to recently visited states, improving feature coverage
  without gradient descent.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .activations import make_activation
from .encoders import make_encoders


class NEFFeatures:
    """Fixed random feature encoder for RL states.

    Encodes a state s ∈ ℝ^d into φ(s) ∈ ℝ^n via::

        φ(s) = activation(gain · (normalize(s) @ Eᵀ) + bias)

    Encoders and biases are fixed at construction time.  State
    normalization uses Welford running statistics (online mean/std).

    Unlike :class:`NEFLayer`, this is a lightweight non-Module class
    optimized for RL use cases: no decoders, no ``nn.Parameter``
    overhead, built-in online normalization.

    Args:
        d_in: observation dimensionality.
        n_neurons: number of encoding neurons.
        activation: activation function name (default ``"abs"``).
        encoder_strategy: encoder generation strategy (default
            ``"hypersphere"``).
        gain: ``(low, high)`` tuple for per-neuron uniform sampling,
            or a scalar for uniform gain.
        centers: optional ``(N, d_in)`` data tensor for data-driven
            biases (``bias = -gain · (center · encoder)``).
        encoder_kwargs: extra keyword arguments forwarded to
            :func:`make_encoders`.
        seed: random seed for reproducibility.
        track_activations: if True, accumulate per-neuron activation
            statistics for dead neuron detection.
    """

    def __init__(
        self,
        d_in: int,
        n_neurons: int,
        activation: str = "abs",
        encoder_strategy: str = "hypersphere",
        gain: float | tuple[float, float] = (0.5, 2.0),
        centers: Tensor | None = None,
        encoder_kwargs: dict | None = None,
        seed: int = 0,
        track_activations: bool = False,
    ):
        rng = torch.Generator().manual_seed(seed)
        self.n_neurons = n_neurons
        self.d_in = d_in

        # Encoders via library factory
        enc_kw = dict(encoder_kwargs or {})
        result = make_encoders(n_neurons, d_in, strategy=encoder_strategy, rng=rng, **enc_kw)
        if isinstance(result, tuple):
            self.encoders, encoder_centers = result[0], result[1]
        else:
            self.encoders = result
            encoder_centers = None

        # Per-neuron gain
        if isinstance(gain, tuple):
            lo, hi = gain
            self._gain = lo + (hi - lo) * torch.rand(n_neurons, generator=rng)
        elif isinstance(gain, (int, float)):
            self._gain = torch.full((n_neurons,), float(gain))
        else:
            self._gain = gain

        # Biases: data-driven from centers or random
        if encoder_centers is not None:
            selected = encoder_centers
        elif centers is not None:
            idx = torch.randint(len(centers), (n_neurons,), generator=rng)
            selected = centers[idx]
        else:
            selected = None

        if selected is not None:
            self.bias = -self._gain * (selected.float() * self.encoders).sum(dim=1)
        else:
            self.bias = (2 * torch.rand(n_neurons, generator=rng) - 1) * self._gain

        # Activation via library factory
        self._act_module = make_activation(activation)

        # Running normalization statistics (Welford online algorithm)
        self._obs_mean = torch.zeros(d_in)
        self._obs_var = torch.ones(d_in)
        self._obs_count = 0

        # Activation tracking for dead neuron detection
        self._track_activations = track_activations
        self._act_sum = torch.zeros(n_neurons)
        self._act_count = 0

    def update_stats(self, obs_batch: Tensor) -> None:
        """Update running mean/variance with a batch of observations."""
        batch_mean = obs_batch.mean(dim=0)
        batch_var = obs_batch.var(dim=0, correction=0)
        batch_count = obs_batch.shape[0]

        total = self._obs_count + batch_count
        if total == 0:
            return
        delta = batch_mean - self._obs_mean
        self._obs_mean = (self._obs_count * self._obs_mean + batch_count * batch_mean) / total
        m_a = self._obs_var * self._obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._obs_count * batch_count / total
        self._obs_var = m2 / total
        self._obs_count = total

    def encode(self, obs: Tensor) -> Tensor:
        """Encode observations to feature vectors.

        Args:
            obs: ``(batch, d_in)`` or ``(d_in,)``

        Returns:
            Features ``(batch, n_neurons)`` or ``(n_neurons,)``.
        """
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)

        std = (self._obs_var + 1e-8).sqrt()
        x = (obs - self._obs_mean) / std

        pre = x @ self.encoders.T
        pre = self._gain * pre + self.bias
        features = self._act_module(pre)

        if self._track_activations:
            self._act_sum += features.detach().sum(dim=0)
            self._act_count += features.shape[0]

        if squeeze:
            features = features.squeeze(0)
        return features

    # -- Activation tracking and recentering ---------------------------------

    def mean_activations(self) -> Tensor:
        """Per-neuron mean activation over tracked samples."""
        if self._act_count == 0:
            return torch.zeros(self.n_neurons)
        return self._act_sum / self._act_count

    def reset_activation_stats(self) -> None:
        """Reset activation tracking counters."""
        self._act_sum.zero_()
        self._act_count = 0

    def dead_neuron_indices(self, percentile: float = 5.0) -> Tensor:
        """Return indices of neurons below the given activation percentile."""
        if self._act_count == 0:
            return torch.tensor([], dtype=torch.long)
        mean_act = self.mean_activations()
        threshold = torch.quantile(mean_act, percentile / 100.0)
        return torch.where(mean_act <= threshold)[0]

    def recenter(self, indices: Tensor, new_centers: Tensor) -> None:
        """Recenter neurons at *indices* to *new_centers*.

        Centers (in raw observation space) are normalized using current
        running statistics before recomputing biases.
        """
        if len(indices) == 0:
            return
        std = (self._obs_var + 1e-8).sqrt()
        normalized = (new_centers.float() - self._obs_mean) / std
        dot = (normalized * self.encoders[indices]).sum(dim=1)
        self.bias[indices] = -self._gain[indices] * dot


class NEFFQIAgent:
    """Fitted Q-Iteration with fixed random NEF features.

    Supports two target modes:

    - **mc**: Monte Carlo returns.  Each solve uses actual discounted
      episode returns G_t = Σ γ^k r_{t+k}.  No bootstrapping, no target
      network, no oscillation risk.
    - **td**: bootstrapped TD targets with Polyak-averaged target
      weights.  Each solve computes y = r + γ(1-d) max Q_target(s').

    Per-action analytical solve::

        w_a = (Φ_aᵀ Φ_a + αI)⁻¹ Φ_aᵀ y_a

    Optional enhancements:

    - **forget_factor** (β): when set, maintains running sufficient
      statistics with exponential decay instead of a replay buffer.
      ``AᵀA ← β·AᵀA + Φ_batch^T Φ_batch``.  Eliminates replay buffer.
    - **recenter_interval**: every N episodes, neurons with activations
      below *recenter_percentile* are recentered to recent observations.

    Args:
        d_obs: observation dimensionality.
        n_actions: number of discrete actions.
        n_neurons: number of encoding neurons.
        gamma: discount factor.
        alpha: regularization strength for Tikhonov solve.
        epsilon_start: initial exploration rate.
        epsilon_end: final exploration rate.
        epsilon_decay: exponential decay constant (episodes).
        buffer_size: maximum replay buffer capacity.
        solve_every: solve frequency in episodes.
        solve_batch: subsample buffer to this size for solve (None = all).
        tau: Polyak averaging coefficient for target weights (TD mode).
        fqi_iters: number of FQI iterations per solve (TD mode).
        target_mode: ``"mc"`` or ``"td"``.
        forget_factor: RLS forgetting factor β ∈ (0, 1].  When set,
            uses exponentially-weighted sufficient statistics instead
            of a replay buffer.  Typical values: 0.99–0.999.
        recenter_interval: if set, check for dead neurons every N
            episodes and recenter them to recently observed states.
        recenter_percentile: activation percentile below which neurons
            are considered dead (default 5.0).
        features_kwargs: extra keyword arguments forwarded to
            :class:`NEFFeatures` (e.g. ``centers``,
            ``encoder_strategy``).
        seed: random seed.
    """

    def __init__(
        self,
        d_obs: int,
        n_actions: int,
        n_neurons: int = 2000,
        gamma: float = 0.99,
        alpha: float = 1e-2,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        buffer_size: int = 50000,
        solve_every: int = 50,
        solve_batch: int | None = None,
        tau: float = 0.3,
        fqi_iters: int = 1,
        target_mode: str = "mc",
        forget_factor: float | None = None,
        recenter_interval: int | None = None,
        recenter_percentile: float = 5.0,
        features_kwargs: dict | None = None,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.n_neurons = n_neurons
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.solve_every = solve_every
        self.solve_batch = solve_batch
        self.tau = tau
        self.fqi_iters = fqi_iters
        self.target_mode = target_mode
        self.forget_factor = forget_factor
        self.recenter_interval = recenter_interval
        self.recenter_percentile = recenter_percentile
        self.rng = np.random.default_rng(seed)

        feat_kw = dict(features_kwargs or {})
        if recenter_interval is not None:
            feat_kw["track_activations"] = True
        self.features = NEFFeatures(d_obs, n_neurons, seed=seed, **feat_kw)

        # Replay buffer (used only in non-RLS mode)
        self.buffer: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._buffer_size = buffer_size
        self._buffer_pos = 0

        # Episode buffer for MC return computation
        self._episode_transitions: list[tuple[np.ndarray, int, float]] = []

        # Current Q-weights and target Q-weights (TD mode only)
        self.W = torch.zeros(n_neurons, n_actions)
        self.W_target = torch.zeros(n_neurons, n_actions)

        # RLS sufficient statistics (per-action, float64)
        if forget_factor is not None:
            self._ata = torch.zeros(n_actions, n_neurons, n_neurons, dtype=torch.float64)
            self._aty = torch.zeros(n_actions, n_neurons, dtype=torch.float64)
            self._n_accumulated = torch.zeros(n_actions, dtype=torch.long)

        # Recent observations for recentering
        if recenter_interval is not None:
            self._recent_obs: deque[np.ndarray] = deque(maxlen=2000)

        self._episode = 0
        self._total_steps = 0
        self._solve_count = 0
        self._recentered_total = 0

    def _epsilon(self) -> float:
        """Current exploration rate (exponential decay)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self._episode / self.epsilon_decay
        )

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using ε-greedy policy."""
        if self.rng.random() < self._epsilon():
            return int(self.rng.integers(self.n_actions))

        with torch.no_grad():
            s = torch.from_numpy(obs).float()
            phi = self.features.encode(s)
            q = phi @ self.W
            return int(q.argmax().item())

    def q_values(self, obs: np.ndarray) -> Tensor:
        """Compute Q-values for all actions (no gradient)."""
        with torch.no_grad():
            s = torch.from_numpy(obs).float()
            phi = self.features.encode(s)
            return phi @ self.W

    def _add_to_buffer(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        transition = (obs, action, reward, next_obs, done)
        if len(self.buffer) < self._buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self._buffer_pos] = transition
        self._buffer_pos = (self._buffer_pos + 1) % self._buffer_size

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Process a single transition (observation normalization + buffer)."""
        self.features.update_stats(torch.from_numpy(np.stack([obs, next_obs])).float())
        self._total_steps += 1

        if self.recenter_interval is not None:
            self._recent_obs.append(obs.copy())

        if self.target_mode == "mc":
            self._episode_transitions.append((obs, action, reward))
            if done:
                self._flush_mc_episode()
        else:
            self._add_to_buffer(obs, action, reward, next_obs, done)

    def _flush_mc_episode(self) -> None:
        """Compute discounted returns and add to buffer or accumulate."""
        transitions = self._episode_transitions
        self._episode_transitions = []
        if not transitions:
            return

        # Compute discounted returns in reverse
        G = 0.0
        episode_data: list[tuple[np.ndarray, int, float]] = []
        for i in reversed(range(len(transitions))):
            obs, action, reward = transitions[i]
            G = reward + self.gamma * G
            episode_data.append((obs, action, G))
        episode_data.reverse()

        if self.forget_factor is not None:
            self._accumulate_episode(episode_data)
        else:
            for obs, action, ret in episode_data:
                self._add_to_buffer(obs, action, ret, obs, True)

    def _accumulate_episode(self, episode_data: list[tuple[np.ndarray, int, float]]) -> None:
        """Accumulate episode into per-action sufficient statistics with decay."""
        states = torch.tensor(np.array([t[0] for t in episode_data]), dtype=torch.float32)
        actions = np.array([t[1] for t in episode_data])
        targets = torch.tensor(np.array([t[2] for t in episode_data]), dtype=torch.float32)

        with torch.no_grad():
            phi = self.features.encode(states)

        phi_f64 = phi.double()
        targets_f64 = targets.double()
        beta = self.forget_factor

        for a in range(self.n_actions):
            mask = actions == a
            if mask.sum() == 0:
                # Still decay even when no samples for this action
                self._ata[a] *= beta
                self._aty[a] *= beta
                continue

            phi_a = phi_f64[mask]
            y_a = targets_f64[mask]

            self._ata[a] = beta * self._ata[a] + phi_a.T @ phi_a
            self._aty[a] = beta * self._aty[a] + phi_a.T @ y_a
            self._n_accumulated[a] += mask.sum()

    def _solve_per_action(self, phi: Tensor, actions: np.ndarray, targets: Tensor) -> Tensor:
        """Per-action analytical solve: w_a = (Φ_aᵀ Φ_a + αI)⁻¹ Φ_aᵀ y."""
        W_new = torch.zeros_like(self.W)
        phi_f64 = phi.double()

        for a in range(self.n_actions):
            mask = actions == a
            if mask.sum() < 10:
                W_new[:, a] = self.W[:, a]
                continue

            phi_a = phi_f64[mask]
            y_a = targets[mask].double()

            ATA = phi_a.T @ phi_a
            ATy = phi_a.T @ y_a

            reg = self.alpha * torch.trace(ATA) / self.n_neurons
            ATA.diagonal().add_(reg.clamp(min=self.alpha))

            try:
                w = torch.linalg.solve(ATA, ATy)
            except torch.linalg.LinAlgError:
                w = torch.linalg.lstsq(phi_a, y_a.unsqueeze(1)).solution.squeeze(1)

            W_new[:, a] = w.float()

        return W_new

    def solve(self) -> None:
        """Solve for Q-weights from replay buffer or RLS statistics.

        In MC mode, targets are discounted returns — direct regression
        with no bootstrapping.  In TD mode, runs FQI iterations with
        target-weight bootstrap.

        When ``forget_factor`` is set, solves from accumulated
        sufficient statistics instead of re-encoding the buffer.
        """
        if self.forget_factor is not None:
            self._solve_from_stats()
        else:
            self._solve_from_buffer()

    def _solve_from_stats(self) -> None:
        """Solve from exponentially-weighted sufficient statistics."""
        W_new = torch.zeros_like(self.W)

        for a in range(self.n_actions):
            if self._n_accumulated[a] < 10:
                W_new[:, a] = self.W[:, a]
                continue

            ATA = self._ata[a].clone()
            ATY = self._aty[a].clone()

            reg = self.alpha * torch.trace(ATA) / self.n_neurons
            ATA.diagonal().add_(reg.clamp(min=self.alpha))

            try:
                w = torch.linalg.solve(ATA, ATY)
            except torch.linalg.LinAlgError:
                w = torch.linalg.lstsq(ATA, ATY.unsqueeze(1)).solution.squeeze(1)

            W_new[:, a] = w.float()

        self.W = W_new

    def _solve_from_buffer(self) -> None:
        """Solve from replay buffer (original method)."""
        if len(self.buffer) < 100:
            return

        n = len(self.buffer)
        if self.solve_batch is not None and self.solve_batch < n:
            idx = self.rng.integers(n, size=self.solve_batch)
            batch = [self.buffer[i] for i in idx]
        else:
            batch = list(self.buffer)

        states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32)
        actions = np.array([t[1] for t in batch])
        values = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float32)

        with torch.no_grad():
            phi = self.features.encode(states)

        if self.target_mode == "mc":
            self.W = self._solve_per_action(phi, actions, values)
        else:
            next_states = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32)
            dones = torch.tensor(np.array([t[4] for t in batch]), dtype=torch.float32)
            with torch.no_grad():
                phi_next = self.features.encode(next_states)

            for _ in range(self.fqi_iters):
                with torch.no_grad():
                    q_next = phi_next @ self.W_target
                    max_q_next = q_next.max(dim=1).values
                    targets = values + self.gamma * (1 - dones) * max_q_next
                self.W = self._solve_per_action(phi, actions, targets)

        self._solve_count += 1

    def end_episode(self) -> None:
        """Signal end of episode; triggers periodic solve + optional recentering."""
        self._episode += 1

        # Dead neuron recentering (before solve so new neurons get fresh data)
        if self.recenter_interval is not None and self._episode % self.recenter_interval == 0:
            self._recenter_dead()

        if self._episode % self.solve_every == 0:
            self.solve()
            self.W_target.mul_(1 - self.tau).add_(self.W * self.tau)

    def _recenter_dead(self) -> int:
        """Recenter dead neurons to recently observed states.

        Returns the number of neurons recentered.
        """
        if self.recenter_interval is None:
            return 0
        recent = list(self._recent_obs)
        if len(recent) < 100:
            return 0

        dead = self.features.dead_neuron_indices(self.recenter_percentile)
        if len(dead) == 0:
            self.features.reset_activation_stats()
            return 0

        recent_t = torch.tensor(np.array(recent), dtype=torch.float32)
        idx = torch.randint(len(recent_t), (len(dead),))
        new_centers = recent_t[idx]

        self.features.recenter(dead, new_centers)

        # Reset RLS statistics for recentered neurons
        if self.forget_factor is not None:
            for a in range(self.n_actions):
                self._ata[a][dead, :] = 0
                self._ata[a][:, dead] = 0
                self._aty[a][dead] = 0

        self._recentered_total += len(dead)
        self.features.reset_activation_stats()
        return len(dead)

    def warmup(
        self,
        env_step: Callable[[int], tuple[np.ndarray, float, bool, bool, dict]] | None = None,
        *,
        env=None,
        n_episodes: int = 20,
    ) -> None:
        """Collect transitions with random policy to seed buffer and stats.

        Pass either a gymnasium ``env`` or a callable ``env_step(action)``
        returning ``(next_obs, reward, terminated, truncated, info)``.
        """
        if env is None and env_step is None:
            raise ValueError("Provide either env or env_step")

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = int(self.rng.integers(self.n_actions))
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs

        self.solve()
        self.W_target.copy_(self.W)
