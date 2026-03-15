"""NEFNetwork — multi-layer NEF networks with multiple training strategies."""

import torch
import torch.nn as nn
from torch import Tensor

from .layers import NEFLayer
from .solvers import solve_decoders


def _ce_targets(targets: Tensor) -> Tensor:
    """Convert one-hot targets to class indices for CrossEntropyLoss."""
    if targets.dim() != 2:
        raise ValueError(f"CE loss requires 2-D one-hot targets, got shape {tuple(targets.shape)}")
    return targets.argmax(dim=1)


class NEFNetwork(nn.Module):
    """Multi-layer NEF network.

    Hidden layers apply encode-only (activities serve as inter-layer
    representation).  Only the output layer decodes to the target space.

    Training strategies:
        fit_greedy     — random hidden encoders, analytic output decoders
        fit_hybrid     — alternate analytic decoder solves with gradient
                         encoder updates
        fit_end_to_end — standard SGD with NEF-initialised weights
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_neurons: list[int],
        output_neurons: int = 2000,
        activation: str = "abs",
        encoder_strategy: str = "hypersphere",
        gain: float | tuple[float, float] | Tensor = (0.5, 2.0),
        rng: torch.Generator | None = None,
        centers: Tensor | None = None,
        **act_kwargs,
    ):
        super().__init__()
        self._activation = activation
        self._encoder_strategy = encoder_strategy
        self.hidden = nn.ModuleList()
        prev_dim = d_in
        for i, n in enumerate(hidden_neurons):
            # Only the first layer uses data-driven centers at construction;
            # deeper layers get centers via propagate_centers() after init.
            layer_centers = centers if i == 0 else None
            self.hidden.append(
                NEFLayer(
                    prev_dim,
                    n,
                    1,
                    activation=activation,
                    encoder_strategy=encoder_strategy,
                    trainable_encoders=True,
                    gain=gain,
                    rng=rng,
                    centers=layer_centers,
                    **act_kwargs,
                )
            )
            prev_dim = n
        # Output layer uses centers only when there are no hidden layers
        out_centers = centers if len(hidden_neurons) == 0 else None
        self.output = NEFLayer(
            prev_dim,
            output_neurons,
            d_out,
            activation=activation,
            encoder_strategy=encoder_strategy,
            trainable_encoders=True,
            gain=gain,
            rng=rng,
            centers=out_centers,
            **act_kwargs,
        )

        # Propagate data-driven biases to deeper layers
        if centers is not None and len(hidden_neurons) > 0:
            self.propagate_centers(centers)

    @torch.no_grad()
    def propagate_centers(self, x: Tensor) -> None:
        """Forward *x* through each layer and use activations as centers for
        the next layer, giving all layers data-driven biases."""
        h = x
        for layer in self.hidden:
            layer.set_centers(h)
            h = layer.encode(h)
        self.output.set_centers(h)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.hidden:
            x = layer.encode(x)
        return self.output(x)

    def _encode_hidden(self, x: Tensor) -> Tensor:
        """Forward through hidden layers (encode only), no grad."""
        with torch.no_grad():
            for layer in self.hidden:
                x = layer.encode(x)
        return x

    # ------------------------------------------------------------------
    # Strategy A — greedy: freeze everything, solve output decoders only
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_greedy(
        self, x: Tensor, targets: Tensor, solver: str = "tikhonov", **solver_kw
    ) -> None:
        """Random hidden features → analytic output decoders."""
        h = x
        for layer in self.hidden:
            h = layer.encode(h)
        self.output.fit(h, targets, solver=solver, **solver_kw)

    # ------------------------------------------------------------------
    # Strategy B — hybrid: analytic decoders + gradient encoder updates
    # ------------------------------------------------------------------

    def fit_hybrid(
        self,
        x: Tensor,
        targets: Tensor,
        n_iters: int = 10,
        lr: float = 1e-3,
        solver: str = "tikhonov",
        loss: str = "mse",
        schedule: bool = False,
        init: str = "random",
        batch_size: int | None = None,
        grad_steps: int = 1,
        centers: Tensor | None = None,
        **solver_kw,
    ) -> None:
        """Alternate between analytic decoder solves and encoder gradient steps.

        Each iteration: (1) solve output decoders from current activities,
        (2) backprop loss to update all encoders and biases.

        Args:
            loss: ``"mse"`` or ``"ce"`` (cross-entropy).  Only affects the
                  encoder gradient signal; decoder solve is always
                  least-squares.
            schedule: use cosine-annealing LR schedule over *n_iters*.
            init: ``"random"`` (default) or ``"incremental"`` — solve a
                  single-layer NEF first and copy its encoders into the
                  first hidden layer.
            batch_size: if set, use mini-batch gradient steps instead of
                        full-batch.
            grad_steps: number of gradient steps per decoder solve (only
                        meaningful with *batch_size*).
            centers: training data for incremental init bias derivation
                     (typically the same *x* passed to this method).
        """
        # --- optional incremental initialisation ---
        if init == "incremental" and len(self.hidden) > 0:
            h0 = self.hidden[0]
            tmp = NEFLayer(
                h0.d_in,
                h0.n_neurons,
                self.output.d_out,
                activation=self._activation,
                encoder_strategy=self._encoder_strategy,
                centers=centers,
            )
            tmp.fit(x, targets, solver=solver, **solver_kw)
            h0.encoders.data.copy_(tmp.encoders.data)
            h0.bias.data.copy_(tmp.bias.data)

        # --- loss function ---
        if loss == "ce":
            loss_fn = nn.CrossEntropyLoss()
            grad_targets = _ce_targets(targets)
        else:
            loss_fn = nn.MSELoss()
            grad_targets = targets

        # --- optimiser + optional schedule ---
        enc_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(enc_params, lr=lr)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)
            if schedule
            else None
        )

        # --- optional mini-batch loader ---
        if batch_size is not None:
            dataset = torch.utils.data.TensorDataset(x, grad_targets)
            g = torch.Generator()
            g.manual_seed(0)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, generator=g
            )

        for _ in range(n_iters):
            # Analytic decoder solve (always full-batch, always MSE)
            h = self._encode_hidden(x)
            self.output.fit(h, targets, solver=solver, **solver_kw)

            # Gradient step(s) on encoders / biases
            if batch_size is not None:
                steps = 0
                for xb, yb in loader:
                    pred = self.forward(xb)
                    batch_loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    steps += 1
                    if steps >= grad_steps:
                        break
            else:
                pred = self.forward(x)
                batch_loss = loss_fn(pred, grad_targets)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

        # Final decoder solve after last encoder update
        h = self._encode_hidden(x)
        self.output.fit(h, targets, solver=solver, **solver_kw)

    # ------------------------------------------------------------------
    # Strategy C — end-to-end SGD with NEF initialisation
    # ------------------------------------------------------------------

    def _sgd_train(
        self, x: Tensor, targets: Tensor, n_epochs: int, lr: float, batch_size: int, loss: str
    ) -> None:
        """Shared SGD training loop used by fit_end_to_end and fit_hybrid_e2e."""
        self.output.decoders.requires_grad_(True)
        try:
            optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

            if loss == "ce":
                loss_fn = nn.CrossEntropyLoss()
                train_targets = _ce_targets(targets)
            else:
                loss_fn = nn.MSELoss()
                train_targets = targets

            dataset = torch.utils.data.TensorDataset(x, train_targets)
            g = torch.Generator()
            g.manual_seed(0)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, generator=g
            )

            for _ in range(n_epochs):
                for xb, yb in loader:
                    pred = self.forward(xb)
                    batch_loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                scheduler.step()
        finally:
            self.output.decoders.requires_grad_(False)

    def fit_end_to_end(
        self,
        x: Tensor,
        targets: Tensor,
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        loss: str = "mse",
    ) -> None:
        """Standard SGD on all parameters, initialised via greedy NEF solve.

        Args:
            loss: ``"mse"`` for regression / one-hot targets, or ``"ce"``
                  for cross-entropy (targets should be one-hot; converted
                  to class indices internally).
        """
        self.fit_greedy(x, targets)
        self._sgd_train(x, targets, n_epochs, lr, batch_size, loss)

    # ------------------------------------------------------------------
    # Strategy D — hybrid then end-to-end refinement
    # ------------------------------------------------------------------

    def fit_hybrid_e2e(
        self,
        x: Tensor,
        targets: Tensor,
        n_iters: int = 10,
        hybrid_lr: float = 1e-3,
        solver: str = "tikhonov",
        n_epochs: int = 20,
        e2e_lr: float = 1e-3,
        batch_size: int = 256,
        loss: str = "ce",
        **hybrid_kw,
    ) -> None:
        """Run hybrid training, then refine with end-to-end SGD.

        First phase uses ``fit_hybrid`` to learn good encoder orientations
        with analytic decoders.  Second phase runs full SGD (including
        decoders) from that warm start — no greedy reset.
        """
        self.fit_hybrid(x, targets, n_iters=n_iters, lr=hybrid_lr, solver=solver, **hybrid_kw)
        self._sgd_train(x, targets, n_epochs, e2e_lr, batch_size, loss)

    # ------------------------------------------------------------------
    # Strategy E — analytical target propagation (NEF-TP)
    # ------------------------------------------------------------------

    def fit_target_prop(
        self,
        x: Tensor,
        targets: Tensor,
        n_iters: int = 50,
        lr: float = 1e-3,
        eta: float = 0.1,
        solver: str = "tikhonov",
        schedule: bool = False,
        normalize_step: bool = True,
        **solver_kw,
    ) -> None:
        """Analytical target propagation using NEF representational decoders.

        Each iteration:
          1. Forward pass with gradient tracking — collect all layer
             inputs and activities (single encode, reused for local loss).
          2. Solve output task decoder analytically.
          3. Solve representational decoders for layers 1..L (layer 0's
             repr decoder is never needed by DTP backward).
          4. Compute output target activities via normalised gradient step:
             ``target = a - eta * grad / ||grad|| * ||a||`` so *eta*
             directly controls the fractional step size.
          5. Propagate targets backward via Difference Target Propagation:
             ``target[l] = a[l] + (target[l+1] - a[l+1]) @ D_repr[l+1]``
          6. Update each layer's encoders and biases with a local gradient
             step minimising ``||encode(input) - target||²``.
          7. Re-solve output decoders after all iterations.

        No backpropagation through multiple layers is needed — all encoder
        updates use single-layer gradients only.

        Args:
            n_iters:        number of TP iterations (default 50).
            lr:             learning rate for local encoder updates.
            eta:            step size for output target computation.
                            With *normalize_step=True* (default), this is
                            the fraction of activity norm to step by.
            solver:         decoder solver (default ``"tikhonov"``).
            schedule:       use cosine-annealing LR schedule over *n_iters*.
            normalize_step: normalise the output gradient so *eta* controls
                            the step as a fraction of activity norm (default
                            True).  When False, raw (unscaled) gradient is
                            used, matching the original formulation.
        """
        all_layers = list(self.hidden) + [self.output]
        n_layers = len(all_layers)

        # One optimizer per layer for independent local updates
        optimizers = []
        for layer in all_layers:
            params = [p for p in layer.parameters() if p.requires_grad]
            optimizers.append(torch.optim.Adam(params, lr=lr))

        schedulers = None
        if schedule:
            schedulers = [
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters)
                for opt in optimizers
            ]

        for _ in range(n_iters):
            # ── Forward pass WITH grad, detach at layer boundaries ─
            # Each layer's activities are connected to its own encoder
            # params only, so we can reuse them for the local loss
            # without redundant re-encoding.
            inputs = [x]
            activities = []
            h = x
            for idx, layer in enumerate(all_layers):
                if idx > 0:
                    h = h.detach()
                    inputs.append(h)
                a = layer.encode(h)
                activities.append(a)
                h = a

            # ── Factorize output Gram matrix once (Cholesky) ──────
            # The task decoder and the output-layer repr decoder both
            # use a_out, so we compute A^T A and its Cholesky factor
            # once and reuse it for both solves.
            a_out = activities[-1].detach()
            AT_out = a_out.T
            ATA = AT_out @ a_out
            alpha = solver_kw.get("alpha", 1e-2)
            reg = alpha * torch.trace(ATA) / ATA.shape[0]
            ATA.diagonal().add_(reg.clamp(min=alpha))
            L_out = torch.linalg.cholesky(ATA)

            D_out = torch.cholesky_solve(AT_out @ targets, L_out)
            self.output.decoders.data.copy_(D_out)

            # ── Solve representational decoders for layers 1..L ───
            # Layer 0's repr decoder is never used by DTP backward,
            # so skip it entirely.  The output layer reuses L_out.
            repr_decoders: list[Tensor | None] = [None]
            for idx in range(1, n_layers):
                if idx == n_layers - 1:
                    D_repr = torch.cholesky_solve(AT_out @ inputs[idx], L_out)
                else:
                    D_repr = solve_decoders(
                        activities[idx].detach(),
                        inputs[idx],
                        method=solver,
                        **solver_kw,
                    )
                repr_decoders.append(D_repr)

            # ── Compute output target activities ──────────────────
            error = a_out @ D_out - targets
            grad = error @ D_out.T
            if normalize_step:
                grad_norm = grad.norm()
                if grad_norm > 0:
                    grad = grad / grad_norm * a_out.norm()
            target_out = a_out - eta * grad
            layer_targets: list[Tensor | None] = [None] * n_layers
            layer_targets[-1] = target_out

            # ── DTP backward pass ─────────────────────────────────
            for idx in range(n_layers - 2, -1, -1):
                a_det = activities[idx + 1].detach()
                delta = layer_targets[idx + 1] - a_det
                layer_targets[idx] = activities[idx].detach() + delta @ repr_decoders[idx + 1]

            # ── Local encoder updates (reuse forward-pass graph) ──
            for idx in range(n_layers):
                target = layer_targets[idx].detach()
                local_loss = (activities[idx] - target).pow(2).mean()
                optimizers[idx].zero_grad()
                local_loss.backward()
                optimizers[idx].step()

            if schedulers is not None:
                for s in schedulers:
                    s.step()

        # Final decoder solve after last encoder update
        h = self._encode_hidden(x)
        self.output.fit(h, targets, solver=solver, **solver_kw)
