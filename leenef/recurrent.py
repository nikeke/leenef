"""RecurrentNEFLayer — temporal NEF with decode-then-re-encode feedback."""

import torch
import torch.nn as nn
from torch import Tensor

from .activations import make_activation
from .encoders import make_encoders
from .layers import _make_gain
from .solvers import solve_decoders, solve_from_normal_equations

_SUPPORTED_TP_SOLVERS = {"tikhonov", "cholesky"}


def _temporal_projection_matrix(state_decoders: Tensor, encoders: Tensor, d_in: int) -> Tensor:
    """Approximate temporal backward map through the state-feedback channels only."""
    return state_decoders @ encoders[:, d_in:].T


class RecurrentNEFLayer(nn.Module):
    """Recurrent NEF layer implementing the canonical decode-then-re-encode loop.

    At each timestep *t*, the layer:

    1. Concatenates the external input ``u(t)`` with a decoded state
       feedback ``s(t-1)``.
    2. Encodes the augmented input into neuron activities.
    3. Decodes two quantities from the activities:

       - **State** ``s(t) = a(t) @ D_state`` — the representational
         decoder (what the population "thinks" the input was).
       - **Output** ``y(t) = a(t) @ D_out`` — the transformational
         decoder (the task prediction).

    The state decoder closes the recurrent loop: its output is fed back
    to the encoders at the next timestep.  This is the NEF principle of
    Eliasmith and Anderson applied to temporal sequences.

    Args:
        d_in:       Dimension of external input per timestep.
        n_neurons:  Number of neurons in the population.
        d_out:      Dimension of the task output.
        d_state:    Dimension of the recurrent state feedback.
                    Defaults to *d_in* (state carries as much info
                    as one input frame).
        activation: Name of the activation function (default ``"relu"``).
                    Uses ``relu`` rather than ``abs`` because ``abs``
                    has gradient ±1 everywhere, causing gradient
                    explosion through BPTT over many timesteps.
        encoder_strategy: Encoder generation strategy.
        gain:       Neuron gain — scalar, ``(low, high)`` tuple, or
                    ``Tensor(n_neurons,)``.
        centers:    Optional ``(N, T, d_in)`` training sequences for
                    data-driven biases.  The first timestep is used with
                    zero state to compute biases.
        rng:        Optional ``torch.Generator`` for reproducibility.
    """

    def __init__(
        self,
        d_in: int,
        n_neurons: int,
        d_out: int,
        d_state: int | None = None,
        activation: str = "relu",
        encoder_strategy: str = "hypersphere",
        gain: float | tuple[float, float] | Tensor = (0.5, 2.0),
        centers: Tensor | None = None,
        rng: torch.Generator | None = None,
        **act_kwargs,
    ):
        super().__init__()
        if d_state is not None and d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        self.d_in = d_in
        self.n_neurons = n_neurons
        self.d_out = d_out
        self.d_state = d_state if d_state is not None else d_in
        self._rng = rng

        # Per-neuron gain
        self.register_buffer("_gain", _make_gain(gain, n_neurons, rng))

        # Encoders see [u(t), s(t-1)]
        enc = make_encoders(n_neurons, d_in + self.d_state, strategy=encoder_strategy, rng=rng)
        self.encoders = nn.Parameter(enc)

        # Biases — data-driven (from first timestep + zero state) or random
        if centers is not None:
            if centers.dim() == 3:
                first_frames = centers[:, 0, :]
            else:
                first_frames = centers
            aug = torch.cat(
                [first_frames, first_frames.new_zeros(first_frames.shape[0], self.d_state)], dim=-1
            )
            idx = torch.randint(len(aug), (n_neurons,), generator=rng)
            bias = -self._gain * (aug[idx].float() * enc).sum(dim=1)
        else:
            bias = torch.randn(n_neurons, generator=rng)
        self.bias = nn.Parameter(bias)

        self.activation = make_activation(activation, **act_kwargs)

        # State (representational) decoder — learnable in hybrid/E2E
        self.state_decoders = nn.Parameter(
            torch.zeros(n_neurons, self.d_state), requires_grad=False
        )
        # Output (transformational) decoder
        self.decoders = nn.Parameter(torch.zeros(n_neurons, d_out), requires_grad=False)

    @property
    def gain(self) -> Tensor:
        """Per-neuron gain vector (n_neurons,)."""
        return self._gain

    def _validate_sequence(self, seq: Tensor) -> tuple[int, int, int]:
        """Validate a sequence tensor and return its shape."""
        if seq.dim() != 3 or seq.shape[2] != self.d_in:
            raise ValueError(f"Expected seq shape (B, T, {self.d_in}), got {tuple(seq.shape)}")
        B, T, d_in = seq.shape
        if T < 1:
            raise ValueError("Expected seq to contain at least one timestep")
        return B, T, d_in

    def _validate_targets(self, targets: Tensor, batch_size: int) -> None:
        """Validate training targets against the configured output shape."""
        if targets.dim() != 2 or targets.shape != (batch_size, self.d_out):
            raise ValueError(
                f"Expected targets shape ({batch_size}, {self.d_out}), got {tuple(targets.shape)}"
            )

    def _validate_training_data(self, seq: Tensor, targets: Tensor) -> tuple[int, int, int]:
        """Validate sequence/target batches used by fit_* methods."""
        B, T, d_in = self._validate_sequence(seq)
        self._validate_targets(targets, B)
        return B, T, d_in

    def _state_target(self, frame: Tensor) -> Tensor:
        """Current reconstruction-style state target used by non-E2E methods."""
        if self.d_state <= self.d_in:
            return frame[:, : self.d_state]
        target = frame.new_zeros(frame.shape[0], self.d_state)
        target[:, : self.d_in] = frame
        return target

    def encode_step(self, u: Tensor, s_prev: Tensor) -> Tensor:
        """Single-step encode: activities from input + state feedback.

        Args:
            u:      (B, d_in) — external input at this timestep.
            s_prev: (B, d_state) — decoded state from previous step.
        Returns:
            activities: (B, n_neurons)
        """
        if u.dim() != 2 or u.shape[1] != self.d_in:
            raise ValueError(f"Expected u shape (B, {self.d_in}), got {tuple(u.shape)}")
        if s_prev.dim() != 2 or s_prev.shape != (u.shape[0], self.d_state):
            raise ValueError(
                f"Expected s_prev shape ({u.shape[0]}, {self.d_state}), got {tuple(s_prev.shape)}"
            )
        x_aug = torch.cat([u, s_prev], dim=-1)
        return self.activation(self._gain * (x_aug @ self.encoders.T) + self.bias)

    def forward(self, seq: Tensor) -> Tensor:
        """Full forward pass over a sequence.

        Args:
            seq: (B, T, d_in) — input sequence.
        Returns:
            y: (B, d_out) — output decoded from final-step activities.
        """
        B, T, _ = self._validate_sequence(seq)
        s = seq.new_zeros(B, self.d_state)
        for t in range(T):
            a = self.encode_step(seq[:, t], s)
            s = a @ self.state_decoders
        return a @ self.decoders

    # ------------------------------------------------------------------
    # Greedy: fully analytic, iterative
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_greedy(
        self,
        seq: Tensor,
        targets: Tensor,
        n_iters: int = 5,
        solver: str = "tikhonov",
        **solver_kw,
    ) -> None:
        """Iteratively solve state and output decoders analytically.

        Each iteration:
          1. Unroll with current D_state, collecting activities.
          2. Solve D_out from final-step activities → targets.
          3. Solve D_state from all-step activities → inputs
             (representational decoding), using incremental normal
             equations to avoid materialising the full T×B matrix.
          4. Repeat.

        Args:
            seq:      (B, T, d_in) input sequences.
            targets:  (B, d_out) classification/regression targets.
            n_iters:  number of greedy iterations (default 5).
            solver:   solver for D_out (default ``"tikhonov"``).
        """
        B, T, _ = self._validate_training_data(seq, targets)
        alpha = solver_kw.get("alpha", 1e-2)

        for _ in range(n_iters):
            # Unroll and accumulate normal equations for D_state
            AtA = seq.new_zeros(self.n_neurons, self.n_neurons)
            AtY = seq.new_zeros(self.n_neurons, self.d_state)
            s = seq.new_zeros(B, self.d_state)
            final_a = None

            for t in range(T):
                a = self.encode_step(seq[:, t], s)
                AtA.addmm_(a.T, a)
                AtY.addmm_(a.T, self._state_target(seq[:, t]))
                s = a @ self.state_decoders
                final_a = a

            # Solve D_out from final-step activities
            D_out = solve_decoders(final_a, targets, method=solver, **solver_kw)
            self.decoders.data.copy_(D_out)

            # Solve D_state from accumulated normal equations
            D_state = solve_from_normal_equations(AtA, AtY, alpha=alpha)
            self.state_decoders.data.copy_(D_state)

    # ------------------------------------------------------------------
    # Hybrid: analytic D_out + gradient on encoders/bias/D_state
    # ------------------------------------------------------------------

    def fit_hybrid(
        self,
        seq: Tensor,
        targets: Tensor,
        n_iters: int = 10,
        lr: float = 1e-3,
        solver: str = "tikhonov",
        loss: str = "mse",
        schedule: bool = False,
        batch_size: int | None = None,
        grad_steps: int = 1,
        grad_clip: float | None = 1.0,
        **solver_kw,
    ) -> None:
        """Alternate analytic D_out solves with gradient encoder updates.

        The gradient signal flows through the full unrolled sequence
        (BPTT) to update encoders, biases, and D_state.  D_out is
        solved analytically each iteration from the final-step
        activities.

        Args:
            loss: ``"mse"`` or ``"ce"`` (cross-entropy on final output).
            grad_clip: max gradient norm for recurrent BPTT updates.  ``None``
                       disables clipping.
        """
        from .networks import _ce_targets

        self._validate_training_data(seq, targets)

        if loss == "ce":
            loss_fn = nn.CrossEntropyLoss()
            grad_targets = _ce_targets(targets)
        else:
            loss_fn = nn.MSELoss()
            grad_targets = targets

        # D_state participates in gradient updates during hybrid
        self.state_decoders.requires_grad_(True)
        try:
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=lr)
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)
                if schedule
                else None
            )

            if batch_size is not None:
                dataset = torch.utils.data.TensorDataset(seq, grad_targets)
                g = torch.Generator()
                g.manual_seed(0)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, generator=g
                )

            for _ in range(n_iters):
                # Analytic D_out solve (full-batch, no grad)
                with torch.no_grad():
                    B, T, _ = seq.shape
                    s = seq.new_zeros(B, self.d_state)
                    for t in range(T):
                        a = self.encode_step(seq[:, t], s)
                        s = a @ self.state_decoders
                    D_out = solve_decoders(a, targets, method=solver, **solver_kw)
                    self.decoders.data.copy_(D_out)

                # Gradient step(s) on encoders, bias, D_state
                if batch_size is not None:
                    steps = 0
                    for sb, yb in loader:
                        pred = self.forward(sb)
                        batch_loss = loss_fn(pred, yb)
                        optimizer.zero_grad()
                        batch_loss.backward()
                        if grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(params, grad_clip)
                        optimizer.step()
                        steps += 1
                        if steps >= grad_steps:
                            break
                else:
                    pred = self.forward(seq)
                    batch_loss = loss_fn(pred, grad_targets)
                    optimizer.zero_grad()
                    batch_loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            # Final D_out solve
            with torch.no_grad():
                B, T, _ = seq.shape
                s = seq.new_zeros(B, self.d_state)
                for t in range(T):
                    a = self.encode_step(seq[:, t], s)
                    s = a @ self.state_decoders
                D_out = solve_decoders(a, targets, method=solver, **solver_kw)
                self.decoders.data.copy_(D_out)
        finally:
            self.state_decoders.requires_grad_(False)

    # ------------------------------------------------------------------
    # End-to-end: full SGD from greedy init
    # ------------------------------------------------------------------

    def _sgd_train(
        self,
        seq: Tensor,
        targets: Tensor,
        n_epochs: int,
        lr: float,
        batch_size: int,
        loss: str,
        grad_clip: float | None,
    ) -> None:
        """Shared recurrent BPTT loop used by E2E-style strategies."""
        from .networks import _ce_targets

        if loss == "ce":
            loss_fn = nn.CrossEntropyLoss()
            train_targets = _ce_targets(targets)
        else:
            loss_fn = nn.MSELoss()
            train_targets = targets

        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        dataset = torch.utils.data.TensorDataset(seq, train_targets)
        g = torch.Generator()
        g.manual_seed(0)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=g
        )

        for _ in range(n_epochs):
            for sb, yb in loader:
                pred = self.forward(sb)
                batch_loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                batch_loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()
            scheduler.step()

    def fit_end_to_end(
        self,
        seq: Tensor,
        targets: Tensor,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        loss: str = "mse",
        greedy_iters: int = 5,
        grad_clip: float | None = 1.0,
        **greedy_kw,
    ) -> None:
        """Full SGD (BPTT) on all parameters, initialised via greedy solve.

        Args:
            greedy_iters: iterations for the initial greedy solve.
            grad_clip: max gradient norm for recurrent BPTT updates.  ``None``
                       disables clipping.
        """
        self._validate_training_data(seq, targets)
        self.fit_greedy(seq, targets, n_iters=greedy_iters, **greedy_kw)

        self.decoders.requires_grad_(True)
        self.state_decoders.requires_grad_(True)
        try:
            self._sgd_train(seq, targets, n_epochs, lr, batch_size, loss, grad_clip)
        finally:
            self.decoders.requires_grad_(False)
            self.state_decoders.requires_grad_(False)

    def fit_hybrid_e2e(
        self,
        seq: Tensor,
        targets: Tensor,
        n_iters: int = 10,
        hybrid_lr: float = 1e-3,
        solver: str = "tikhonov",
        n_epochs: int = 20,
        e2e_lr: float = 1e-3,
        batch_size: int = 256,
        loss: str = "ce",
        grad_clip: float | None = 1.0,
        **hybrid_kw,
    ) -> None:
        """Run recurrent hybrid training, then refine with end-to-end SGD."""
        self._validate_training_data(seq, targets)
        self.fit_hybrid(
            seq,
            targets,
            n_iters=n_iters,
            lr=hybrid_lr,
            solver=solver,
            loss=loss,
            grad_clip=grad_clip,
            **hybrid_kw,
        )

        self.decoders.requires_grad_(True)
        self.state_decoders.requires_grad_(True)
        try:
            self._sgd_train(seq, targets, n_epochs, e2e_lr, batch_size, loss, grad_clip)
        finally:
            self.decoders.requires_grad_(False)
            self.state_decoders.requires_grad_(False)

    # ------------------------------------------------------------------
    # Target propagation through time (TPTT)
    # ------------------------------------------------------------------

    def fit_target_prop(
        self,
        seq: Tensor,
        targets: Tensor,
        n_iters: int = 50,
        lr: float = 1e-3,
        eta: float = 0.1,
        solver: str = "tikhonov",
        schedule: bool = False,
        normalize_step: bool = True,
        batch_size: int | None = None,
        **solver_kw,
    ) -> None:
        """Target propagation through time using NEF state decoders as inverse models.

        Instead of BPTT, targets are propagated backward through timesteps:
        the output-layer target is computed from a normalised gradient step,
        then difference target propagation uses the state decoder to map
        each timestep's target back to the previous timestep's activity
        space.  Encoder/bias updates use single-timestep gradients only.

        Each iteration has two phases:

        **Phase 1** (no grad, chunked normal equations):
          1. Unroll forward, accumulating ``A^T A`` and ``A^T Y`` for both
             the output and representational decoder solves.
          2. Solve ``D_out`` and ``D_state_repr`` analytically.
          3. Compute final-timestep targets with full-batch normalisation.

        **Phase 2** (with grad, chunked encoder updates):
          4. Re-unroll forward with gradient tracking in memory-safe chunks.
          5. Propagate targets backward through time via DTP within each
             chunk (targets are per-sample, so chunking is exact).
          6. Accumulate local encoder/bias gradients across all chunks,
             then step the optimiser once.

        Re-solve ``D_out`` after all iterations.

        Args:
            n_iters:        number of TP iterations (default 50).
            lr:             learning rate for local encoder updates.
            eta:            step size for output target computation.
            solver:         decoder solver for the output solve.  Recurrent TP
                            currently supports ``"tikhonov"`` and
                            ``"cholesky"`` because it relies on accumulated
                            normal equations.
            schedule:       use cosine-annealing LR schedule.
            normalize_step: normalise the output gradient so *eta* controls
                            the step as a fraction of activity norm.
            batch_size:     chunk size for the gradient phase.  When
                            ``None`` (default), auto-sized to keep peak
                            memory under ~4 GB.
        """
        if solver not in _SUPPORTED_TP_SOLVERS:
            raise ValueError(
                f"fit_target_prop supports only {sorted(_SUPPORTED_TP_SOLVERS)} "
                f"because it uses accumulated normal equations, got {solver!r}"
            )

        B, T, d_in = self._validate_training_data(seq, targets)
        alpha = solver_kw.get("alpha", 1e-2)

        if batch_size is None:
            # ~4 floats per neuron per timestep (activities + targets + autograd)
            bytes_per_sample = T * self.n_neurons * 4 * 4
            batch_size = max(256, min(B, (4 * 1024**3) // max(1, bytes_per_sample)))

        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler_obj = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)
            if schedule
            else None
        )

        for _ in range(n_iters):
            # ── Phase 1: Forward (no grad), accumulate normal equations,
            #    collect final-step activities for target computation ──
            AtA_out = seq.new_zeros(self.n_neurons, self.n_neurons)
            AtY_out = seq.new_zeros(self.n_neurons, self.d_out)
            AtA_state = seq.new_zeros(self.n_neurons, self.n_neurons)
            AtY_state = seq.new_zeros(self.n_neurons, self.d_state)
            a_final_chunks = []

            with torch.no_grad():
                for i in range(0, B, batch_size):
                    sb = seq[i : i + batch_size]
                    tb = targets[i : i + batch_size]
                    s = sb.new_zeros(sb.shape[0], self.d_state)
                    for t in range(T):
                        x_aug = torch.cat([sb[:, t], s], dim=-1)
                        a = self.activation(self._gain * (x_aug @ self.encoders.T) + self.bias)
                        AtA_state.addmm_(a.T, a)
                        AtY_state.addmm_(a.T, self._state_target(sb[:, t]))
                        s = a @ self.state_decoders
                    AtA_out.addmm_(a.T, a)
                    AtY_out.addmm_(a.T, tb)
                    a_final_chunks.append(a)

            # Solve decoders
            D_out = solve_from_normal_equations(AtA_out, AtY_out, alpha=alpha)
            self.decoders.data.copy_(D_out)
            D_state = solve_from_normal_equations(AtA_state, AtY_state, alpha=alpha)
            self.state_decoders.data.copy_(D_state)

            # Compute final-timestep targets (full-batch normalisation)
            a_final_all = torch.cat(a_final_chunks, dim=0)
            del a_final_chunks
            error = a_final_all @ D_out - targets
            grad_out = error @ D_out.T
            if normalize_step:
                gn = grad_out.norm()
                if gn > 0:
                    grad_out = grad_out / gn * a_final_all.norm()
            target_final_all = a_final_all - eta * grad_out
            del a_final_all, error, grad_out

            # Precompute DTP projection matrix (n_neurons × n_neurons)
            DRE = _temporal_projection_matrix(
                self.state_decoders.detach(), self.encoders.detach(), d_in
            )

            # ── Phase 2: Re-forward (with grad) in chunks, DTP targets,
            #    accumulate encoder/bias gradients ──
            optimizer.zero_grad()
            for i in range(0, B, batch_size):
                sb = seq[i : i + batch_size]
                Bb = sb.shape[0]

                # Forward with grad
                activities = []
                s = sb.new_zeros(Bb, self.d_state)
                for t in range(T):
                    x_aug = torch.cat([sb[:, t], s.detach()], dim=-1)
                    a = self.activation(self._gain * (x_aug @ self.encoders.T) + self.bias)
                    activities.append(a)
                    s = a @ self.state_decoders.detach()

                # DTP backward through time
                timestep_targets = [None] * T
                timestep_targets[-1] = target_final_all[i : i + Bb]
                for t in range(T - 2, -1, -1):
                    a_next_det = activities[t + 1].detach()
                    delta = timestep_targets[t + 1] - a_next_det
                    timestep_targets[t] = activities[t].detach() + delta @ DRE

                # Local loss (scaled for correct gradient accumulation)
                chunk_loss = sum(
                    (activities[t] - timestep_targets[t].detach()).pow(2).mean() for t in range(T)
                ) * (Bb / B)
                chunk_loss.backward()

                del activities, timestep_targets, chunk_loss

            optimizer.step()
            del target_final_all

            if scheduler_obj is not None:
                scheduler_obj.step()

        # ── Final D_out solve after last encoder update ──
        with torch.no_grad():
            AtA = seq.new_zeros(self.n_neurons, self.n_neurons)
            AtY = seq.new_zeros(self.n_neurons, self.d_out)
            for i in range(0, B, batch_size):
                sb = seq[i : i + batch_size]
                tb = targets[i : i + batch_size]
                s = sb.new_zeros(sb.shape[0], self.d_state)
                for t in range(T):
                    a = self.encode_step(sb[:, t], s)
                    s = a @ self.state_decoders
                AtA.addmm_(a.T, a)
                AtY.addmm_(a.T, tb)
            D_out = solve_from_normal_equations(AtA, AtY, alpha=alpha)
            self.decoders.data.copy_(D_out)
