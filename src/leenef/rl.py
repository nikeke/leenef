"""NEF-based reinforcement learning components.

Provides :class:`NEFFeatures` (fixed random feature encoder with online
state normalization), :class:`NEFFQIAgent` (Fitted Q-Iteration with
analytical least-squares solve), and :class:`NEFFQIEnsemble` (ensemble
of agents for exploration via disagreement / Thompson sampling).

The core idea: replace DQN's gradient-based Q-network updates with NEF's
analytical solve.  Both approaches approximate Q(s, a); the difference
is *how* the approximation weights are updated — SGD mini-batches vs.
one-shot least-squares on the full replay buffer.

Five target modes are supported:

- **mc** (Monte Carlo): regression targets are discounted episode
  returns G_t = Σ γ^k r_{t+k}.  No bootstrapping, no target network.
  Optionally blended with TD via ``td_lambda`` (λ-returns).
- **td** (Temporal Difference): bootstrapped targets y = r + γ max Q(s').
  Requires a Polyak-averaged target weight matrix for stability.
- **nstep**: truncated n-step returns G_t^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k}.
  Like MC but with a fixed horizon, reducing variance at the cost of
  slight bias.  No bootstrapping.
- **eligibility**: streaming eligibility traces that compute MC returns
  forward.  At each step, e_a ← γ·e_a + Φ(s), AᵀY += e·r.
  Mathematically equivalent to MC but constant memory (no episode
  buffer) and supports mid-episode solves.  Requires ``forget_factor``.
- **differential**: eligibility traces with EMA reward baseline.
  Target is r_t − r̄ (above-average reward credit).  Fully online,
  no episode-boundary dependency.  Requires ``forget_factor``.

Optional mechanisms improve performance:

- **Exponential forgetting** (RLS): sufficient statistics decay as
  AᵀA ← β·AᵀA + batch, eliminating the replay buffer while adapting
  to non-stationary targets.
- **Dead neuron recentering**: neurons with low mean activation are
  recentered to recently visited states, improving feature coverage
  without gradient descent.
- **EMA activity tracking**: exponentially-decayed per-neuron activity
  for smoother dead neuron detection (``activity_decay`` parameter).
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
        activity_decay: if set, use exponentially-decayed activity
            tracking instead of cumulative mean.  Typical values
            0.99–0.999.  Enables smoother dead neuron detection that
            adapts to changing feature usage over time.
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
        activity_decay: float | None = None,
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
        self._activity_decay = activity_decay
        if activity_decay is not None:
            self._ema_activity = torch.zeros(n_neurons)
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
            if self._activity_decay is not None:
                batch_mean = features.detach().mean(dim=0)
                self._ema_activity = (
                    self._activity_decay * self._ema_activity
                    + (1 - self._activity_decay) * batch_mean
                )
            else:
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

    def reset_activation_stats(self, indices: Tensor | None = None) -> None:
        """Reset activation tracking counters.

        Parameters
        ----------
        indices : Tensor or None
            If given, reset only those neurons.  If *None*, reset all.
            For EMA mode, reset neurons get the current global mean
            (fair restart).  For cumulative mode, reset neurons get
            zero sums (the sample count is not changed so surviving
            neurons keep their long-term statistics).
        """
        if indices is not None and len(indices) == 0:
            return
        if self._activity_decay is not None:
            mean_val = self._ema_activity.mean().item()
            if indices is None:
                self._ema_activity.fill_(mean_val)
            else:
                self._ema_activity[indices] = mean_val
        else:
            if indices is None:
                self._act_sum.zero_()
                self._act_count = 0
            else:
                # Give recentered neurons a fair start at the current
                # mean so they aren't immediately flagged as dead again.
                self._act_sum[indices] = self._act_sum.mean()

    def dead_neuron_indices(self, percentile: float = 5.0, min_relative: float = 0.1) -> Tensor:
        """Return indices of genuinely inactive neurons.

        A neuron is "dead" only if its activity is both below the given
        *percentile* of all neurons AND below *min_relative* × the
        population mean.  The absolute floor prevents recentering when
        all neurons are similarly active.

        Uses EMA activity when ``activity_decay`` is set, otherwise
        uses cumulative mean activations.
        """
        if self._activity_decay is not None:
            act = self._ema_activity
            if act.sum() == 0:
                return torch.tensor([], dtype=torch.long)
        else:
            if self._act_count == 0:
                return torch.tensor([], dtype=torch.long)
            act = self.mean_activations()

        pctl_threshold = torch.quantile(act, percentile / 100.0)
        abs_threshold = act.mean() * min_relative
        threshold = min(pctl_threshold.item(), abs_threshold.item())
        return torch.where(act <= threshold)[0]

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

    Supports five target modes:

    - **mc**: Monte Carlo returns.  Each solve uses actual discounted
      episode returns G_t = Σ γ^k r_{t+k}.  When ``td_lambda`` is set
      to a value < 1, computes λ-returns that blend MC with TD
      bootstrapping for lower variance: G_t^λ = r_t + γ[(1-λ)maxQ(s')
      + λ G_{t+1}^λ].
    - **td**: bootstrapped TD targets with Polyak-averaged target
      weights.  Each solve computes y = r + γ(1-d) max Q_target(s').
    - **nstep**: truncated n-step returns (no bootstrapping).
    - **eligibility**: streaming MC via eligibility traces.
      Requires ``forget_factor``.
    - **differential**: eligibility traces + EMA reward baseline.
      Requires ``forget_factor``.

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
        target_mode: ``"mc"``, ``"td"``, ``"nstep"``, ``"eligibility"``,
            or ``"differential"``.
        td_lambda: λ for λ-returns in MC mode.  ``None`` or ``1.0``
            gives pure MC; ``0.0`` gives one-step TD.  Typical values
            0.8–0.95 reduce MC variance with minimal bias.
        forget_factor: RLS forgetting factor β ∈ (0, 1].  When set,
            uses exponentially-weighted sufficient statistics instead
            of a replay buffer.  Typical values: 0.99–0.999.
        n_step: horizon for n-step returns (default 20).
        reward_ema_decay: EMA decay for reward baseline in differential
            mode (default 0.99).
        recenter_interval: if set, check for dead neurons every N
            episodes and recenter them to recently observed states.
        recenter_percentile: activation percentile below which neurons
            are considered dead (default 5.0).
        activity_decay: if set, use EMA activity tracking for
            recentering instead of cumulative mean.
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
        td_lambda: float | None = None,
        forget_factor: float | None = None,
        n_step: int = 20,
        reward_ema_decay: float = 0.99,
        recenter_interval: int | None = None,
        recenter_percentile: float = 5.0,
        activity_decay: float | None = None,
        features_kwargs: dict | None = None,
        seed: int = 0,
    ):
        valid_modes = ("mc", "td", "nstep", "eligibility", "differential")
        if target_mode not in valid_modes:
            raise ValueError(f"target_mode must be one of {valid_modes}, got {target_mode!r}")
        if target_mode in ("eligibility", "differential") and forget_factor is None:
            raise ValueError(f"target_mode={target_mode!r} requires forget_factor")

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
        self.td_lambda = td_lambda
        self.forget_factor = forget_factor
        self.n_step = n_step
        self.reward_ema_decay = reward_ema_decay
        self.recenter_interval = recenter_interval
        self.recenter_percentile = recenter_percentile
        self.rng = np.random.default_rng(seed)

        feat_kw = dict(features_kwargs or {})
        if recenter_interval is not None:
            feat_kw["track_activations"] = True
            if activity_decay is not None:
                feat_kw["activity_decay"] = activity_decay
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

        # Eligibility traces for streaming modes
        if target_mode in ("eligibility", "differential"):
            self._eligibility = torch.zeros(n_actions, n_neurons, dtype=torch.float64)
            self._aty_ep = torch.zeros(n_actions, n_neurons, dtype=torch.float64)
            self._ep_features: list[tuple[Tensor, int]] = []
        if target_mode == "differential":
            self._reward_ema = 0.0

        self._episode = 0
        self._total_steps = 0
        self._solve_count = 0
        self._recentered_total = 0

    @property
    def _use_lambda_returns(self) -> bool:
        """True when λ-returns should be computed instead of pure MC."""
        return self.td_lambda is not None and self.td_lambda < 1.0

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

        if self.target_mode in ("eligibility", "differential"):
            self._step_accumulate(obs, action, reward, done)
        elif self.target_mode in ("mc", "nstep"):
            if self.target_mode == "mc" and self._use_lambda_returns:
                self._episode_transitions.append((obs, action, reward, next_obs))
            else:
                self._episode_transitions.append((obs, action, reward))
            if done:
                self._flush_mc_episode()
        else:
            self._add_to_buffer(obs, action, reward, next_obs, done)

    def _flush_mc_episode(self) -> None:
        """Compute discounted returns (or λ-returns) and add to buffer or accumulate."""
        transitions = self._episode_transitions
        self._episode_transitions = []
        if not transitions:
            return

        if self.target_mode == "nstep":
            episode_data = self._compute_nstep_returns(transitions)
        elif self._use_lambda_returns:
            episode_data = self._compute_lambda_returns(transitions)
        else:
            episode_data = self._compute_mc_returns(transitions)

        if self.forget_factor is not None:
            self._accumulate_episode(episode_data)
        else:
            for obs, action, ret in episode_data:
                self._add_to_buffer(obs, action, ret, obs, True)

    def _compute_mc_returns(
        self,
        transitions: list[tuple],
    ) -> list[tuple[np.ndarray, int, float]]:
        """Pure Monte Carlo returns."""
        G = 0.0
        episode_data: list[tuple[np.ndarray, int, float]] = []
        for i in reversed(range(len(transitions))):
            obs, action, reward = transitions[i][:3]
            G = reward + self.gamma * G
            episode_data.append((obs, action, G))
        episode_data.reverse()
        return episode_data

    def _compute_nstep_returns(
        self,
        transitions: list[tuple],
    ) -> list[tuple[np.ndarray, int, float]]:
        """Truncated n-step returns (no bootstrapping).

        G_t^(n) = Σ_{k=0}^{min(n,T-t)-1} γ^k r_{t+k}

        Like MC but capped at n steps.  Reduces variance at the cost
        of ignoring rewards beyond the horizon.
        """
        T = len(transitions)
        n = self.n_step
        rewards = [t[2] for t in transitions]
        episode_data: list[tuple[np.ndarray, int, float]] = []
        for t in range(T):
            G = 0.0
            horizon = min(n, T - t)
            for k in reversed(range(horizon)):
                G = rewards[t + k] + self.gamma * G
            obs, action = transitions[t][0], transitions[t][1]
            episode_data.append((obs, action, G))
        return episode_data

    def _step_accumulate(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """Per-step accumulation for eligibility/differential modes.

        For eligibility mode, this is mathematically equivalent to MC
        returns but computed forward via eligibility traces:

            e_a ← γ · e_a + Φ(s)    (for taken action a)
            e_b ← γ · e_b           (for other actions b)
            AᵀY_a += e_a · r        (for all actions)

        The sum Σ_t e_t · r_t = Σ_k Φ(s_k) · G_k exactly reproduces
        MC returns without needing to store episode transitions or
        compute backward.

        For differential mode, uses r − r̄ instead of r, where r̄ is
        an EMA reward baseline.

        AᵀA is deferred to episode end (batch matmul) for performance.
        Per-step eligibility and AᵀY updates are O(n_neurons × n_actions).
        """
        phi = self.features.encode(torch.from_numpy(obs).float()).detach()
        phi_f64 = phi.double()

        # Buffer features for batch AᵀA at episode end
        self._ep_features.append((phi_f64, action))

        # Decay all eligibility traces, add feature for taken action
        self._eligibility *= self.gamma
        self._eligibility[action] += phi_f64

        # Compute target
        if self.target_mode == "differential":
            alpha = 1 - self.reward_ema_decay
            self._reward_ema = self.reward_ema_decay * self._reward_ema + alpha * reward
            target = reward - self._reward_ema
        else:
            target = reward

        # Accumulate eligibility-weighted target for ALL actions (O(n × n_actions))
        for a in range(self.n_actions):
            self._aty_ep[a] += self._eligibility[a] * target

        if done:
            self._flush_eligibility_episode()

    def _flush_eligibility_episode(self) -> None:
        """Batch-process deferred AᵀA and merge episode stats."""
        features_data = self._ep_features
        self._ep_features = []

        if not features_data:
            return

        beta = self.forget_factor

        # Batch AᵀA computation (like MC mode)
        actions_arr = np.array([a for _, a in features_data])
        phi_all = torch.stack([phi for phi, _ in features_data])

        for a in range(self.n_actions):
            mask = actions_arr == a
            self._ata[a] *= beta
            if mask.sum() > 0:
                phi_a = phi_all[mask]
                self._ata[a] += phi_a.T @ phi_a
            self._n_accumulated[a] += int(mask.sum())

        # Merge per-episode AᵀY into global with decay
        for a in range(self.n_actions):
            self._aty[a] = beta * self._aty[a] + self._aty_ep[a]
            self._aty_ep[a].zero_()

        # Reset eligibility traces
        self._eligibility.zero_()

    def _compute_lambda_returns(
        self,
        transitions: list[tuple],
    ) -> list[tuple[np.ndarray, int, float]]:
        """λ-returns blending MC and TD bootstrapping.

        G_t^λ = r_t + γ [(1-λ) max_a Q(s_{t+1}, a) + λ G_{t+1}^λ]

        For the terminal step (last in episode), G_T = r_T (no bootstrap).
        λ=1 recovers pure MC; λ=0 gives one-step TD with current Q.
        """
        lam = self.td_lambda
        T = len(transitions)

        # Bootstrap values: Q(s_{t+1}) for each intermediate step
        next_states = torch.tensor(
            np.array([t[3] for t in transitions]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            phi_next = self.features.encode(next_states)
            q_next = phi_next @ self.W
            max_q = q_next.max(dim=1).values.numpy()

        G = 0.0
        episode_data: list[tuple[np.ndarray, int, float]] = []
        for i in reversed(range(T)):
            obs, action, reward = transitions[i][:3]
            if i == T - 1:
                # Terminal step — no bootstrapping
                G = reward
            else:
                # Blend MC continuation with TD bootstrap
                G = reward + self.gamma * ((1 - lam) * max_q[i] + lam * G)
            episode_data.append((obs, action, G))
        episode_data.reverse()
        return episode_data

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

        # Reset activity stats only for recentered neurons so surviving
        # neurons keep their long-term statistics.
        self.features.reset_activation_stats(dead)
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


class NEFFQIEnsemble:
    """Ensemble of NEFFQIAgents for exploration via disagreement.

    Each member uses different random features (seeds) but trains on the
    same experience.  Diverse projections produce diverse Q estimates;
    disagreement signals uncertainty and drives exploration.

    Three exploration strategies:

    - **thompson**: at each step, sample a random member and follow its
      greedy policy.  Equivalent to Thompson sampling with the ensemble
      as a posterior approximation.
    - **voting**: majority vote on the greedy action of each member.
      Ties are broken by Thompson sampling (random member).
    - **ucb**: take the action with highest (mean_Q + ucb_coeff * std_Q)
      across members.  Optimistic in the face of uncertainty.

    After the ε-greedy exploration period ends, the ensemble's diversity
    provides ongoing exploration without ε noise.

    Args:
        n_members: number of ensemble members.
        explore_strategy: ``"thompson"``, ``"voting"``, or ``"ucb"``.
        ucb_coeff: coefficient for UCB exploration bonus (only used
            when ``explore_strategy="ucb"``).
        base_seed: starting seed; member *i* uses ``base_seed + i``.
        **agent_kwargs: forwarded to each :class:`NEFFQIAgent`.
    """

    def __init__(
        self,
        n_members: int = 5,
        explore_strategy: str = "thompson",
        ucb_coeff: float = 1.0,
        base_seed: int = 0,
        **agent_kwargs,
    ):
        if explore_strategy not in ("thompson", "ucb", "voting"):
            raise ValueError(
                f"explore_strategy must be 'thompson', 'voting', or 'ucb', "
                f"got {explore_strategy!r}"
            )
        self.n_members = n_members
        self.explore_strategy = explore_strategy
        self.ucb_coeff = ucb_coeff
        self.rng = np.random.default_rng(base_seed)

        self.members: list[NEFFQIAgent] = []
        for i in range(n_members):
            kw = dict(agent_kwargs)
            kw["seed"] = base_seed + i
            self.members.append(NEFFQIAgent(**kw))

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using ensemble exploration strategy.

        During ε-greedy phase, each member's ε-greedy still applies
        (Thompson picks a random member, then uses its ε-greedy).
        After ε decays, ensemble diversity provides exploration.
        """
        if self.explore_strategy == "thompson":
            idx = int(self.rng.integers(self.n_members))
            return self.members[idx].select_action(obs)

        if self.explore_strategy == "voting":
            votes = [m.select_action(obs) for m in self.members]
            counts: dict[int, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            max_count = max(counts.values())
            winners = [a for a, c in counts.items() if c == max_count]
            if len(winners) == 1:
                return winners[0]
            # Tie-break: Thompson (random member among those that voted for a winner)
            idx = int(self.rng.integers(self.n_members))
            return self.members[idx].select_action(obs)

        # UCB: mean Q + coeff * std Q across members
        q_all = torch.stack([m.q_values(obs) for m in self.members])
        mean_q = q_all.mean(dim=0)
        std_q = q_all.std(dim=0)
        ucb = mean_q + self.ucb_coeff * std_q
        return int(ucb.argmax().item())

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Broadcast transition to all ensemble members."""
        for member in self.members:
            member.update(obs, action, reward, next_obs, done)

    def end_episode(self) -> None:
        """Signal end of episode to all members."""
        for member in self.members:
            member.end_episode()

    def warmup(self, *, env, n_episodes: int = 20) -> None:
        """Warmup all members using random exploration.

        Unlike individual warmup, runs episodes once and broadcasts
        transitions to all members (avoids N× environment interaction).
        """
        n_actions = self.members[0].n_actions
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = int(self.rng.integers(n_actions))
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
            self.end_episode()

        # Force initial solve on all members
        for member in self.members:
            member.solve()
            member.W_target.copy_(member.W)

    def q_values(self, obs: np.ndarray) -> Tensor:
        """Mean Q-values across ensemble members."""
        q_all = torch.stack([m.q_values(obs) for m in self.members])
        return q_all.mean(dim=0)

    @property
    def _episode(self) -> int:
        """Current episode count (from first member)."""
        return self.members[0]._episode

    @property
    def _recentered_total(self) -> int:
        """Total neurons recentered across all members."""
        return sum(m._recentered_total for m in self.members)
