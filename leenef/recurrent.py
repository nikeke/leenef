"""RecurrentNEFLayer — temporal NEF with decode-then-re-encode feedback."""

import torch
import torch.nn as nn
from torch import Tensor

from .activations import make_activation
from .encoders import make_encoders
from .layers import _make_gain
from .solvers import solve_decoders, solve_from_normal_equations


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
        activation: Name of the activation function (default ``"abs"``).
        encoder_strategy: Encoder generation strategy.
        gain:       Neuron gain — scalar, ``(low, high)`` tuple, or
                    ``Tensor(n_neurons,)``.
        rng:        Optional ``torch.Generator`` for reproducibility.
    """

    def __init__(
        self,
        d_in: int,
        n_neurons: int,
        d_out: int,
        d_state: int | None = None,
        activation: str = "abs",
        encoder_strategy: str = "hypersphere",
        gain: float | tuple[float, float] | Tensor = 1.0,
        rng: torch.Generator | None = None,
        **act_kwargs,
    ):
        super().__init__()
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
        self.bias = nn.Parameter(torch.randn(n_neurons, generator=rng))

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

    def encode_step(self, u: Tensor, s_prev: Tensor) -> Tensor:
        """Single-step encode: activities from input + state feedback.

        Args:
            u:      (B, d_in) — external input at this timestep.
            s_prev: (B, d_state) — decoded state from previous step.
        Returns:
            activities: (B, n_neurons)
        """
        x_aug = torch.cat([u, s_prev], dim=-1)
        return self.activation(self._gain * (x_aug @ self.encoders.T) + self.bias)

    def forward(self, seq: Tensor) -> Tensor:
        """Full forward pass over a sequence.

        Args:
            seq: (B, T, d_in) — input sequence.
        Returns:
            y: (B, d_out) — output decoded from final-step activities.
        """
        B, T, _ = seq.shape
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
        B, T, d_in = seq.shape
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
                # State target: reconstruct input (truncated or padded to d_state)
                if self.d_state <= d_in:
                    state_target = seq[:, t, : self.d_state]
                else:
                    state_target = seq.new_zeros(B, self.d_state)
                    state_target[:, :d_in] = seq[:, t]
                AtY.addmm_(a.T, state_target)
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
        **solver_kw,
    ) -> None:
        """Alternate analytic D_out solves with gradient encoder updates.

        The gradient signal flows through the full unrolled sequence
        (BPTT) to update encoders, biases, and D_state.  D_out is
        solved analytically each iteration from the final-step
        activities.

        Args:
            loss: ``"mse"`` or ``"ce"`` (cross-entropy on final output).
        """
        from .networks import _ce_targets

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
                        optimizer.step()
                        steps += 1
                        if steps >= grad_steps:
                            break
                else:
                    pred = self.forward(seq)
                    batch_loss = loss_fn(pred, grad_targets)
                    optimizer.zero_grad()
                    batch_loss.backward()
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

    def fit_end_to_end(
        self,
        seq: Tensor,
        targets: Tensor,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        loss: str = "mse",
        greedy_iters: int = 5,
        **greedy_kw,
    ) -> None:
        """Full SGD (BPTT) on all parameters, initialised via greedy solve.

        Args:
            greedy_iters: iterations for the initial greedy solve.
        """
        from .networks import _ce_targets

        self.fit_greedy(seq, targets, n_iters=greedy_iters, **greedy_kw)

        self.decoders.requires_grad_(True)
        self.state_decoders.requires_grad_(True)
        try:
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
                    optimizer.step()
                scheduler.step()
        finally:
            self.decoders.requires_grad_(False)
            self.state_decoders.requires_grad_(False)
