"""NEFNetwork — multi-layer NEF networks with multiple training strategies."""

import torch
import torch.nn as nn
from torch import Tensor

from .layers import NEFLayer


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

    def __init__(self, d_in: int, d_out: int,
                 hidden_neurons: list[int],
                 output_neurons: int = 2000,
                 activation: str = "relu",
                 encoder_strategy: str = "gaussian",
                 gain: float = 1.0,
                 rng: torch.Generator | None = None,
                 centers: Tensor | None = None,
                 **act_kwargs):
        super().__init__()
        self.hidden = nn.ModuleList()
        prev_dim = d_in
        for i, n in enumerate(hidden_neurons):
            # Only the first layer uses data-driven centers
            layer_centers = centers if i == 0 else None
            self.hidden.append(NEFLayer(
                prev_dim, n, 1,
                activation=activation, encoder_strategy=encoder_strategy,
                trainable_encoders=True, gain=gain, rng=rng,
                centers=layer_centers, **act_kwargs))
            prev_dim = n
        # Output layer uses centers only when there are no hidden layers
        out_centers = centers if len(hidden_neurons) == 0 else None
        self.output = NEFLayer(
            prev_dim, output_neurons, d_out,
            activation=activation, encoder_strategy=encoder_strategy,
            trainable_encoders=True, gain=gain, rng=rng,
            centers=out_centers, **act_kwargs)

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
    def fit_greedy(self, x: Tensor, targets: Tensor,
                   solver: str = "tikhonov", **solver_kw) -> None:
        """Random hidden features → analytic output decoders."""
        h = x
        for layer in self.hidden:
            h = layer.encode(h)
        self.output.fit(h, targets, solver=solver, **solver_kw)

    # ------------------------------------------------------------------
    # Strategy B — hybrid: analytic decoders + gradient encoder updates
    # ------------------------------------------------------------------

    def fit_hybrid(self, x: Tensor, targets: Tensor,
                   n_iters: int = 10, lr: float = 1e-3,
                   solver: str = "tikhonov", **solver_kw) -> None:
        """Alternate between analytic decoder solves and encoder gradient steps.

        Each iteration: (1) solve output decoders from current activities,
        (2) backprop MSE loss to update all encoders and biases.
        """
        enc_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(enc_params, lr=lr)

        for _ in range(n_iters):
            # Analytic decoder solve
            h = self._encode_hidden(x)
            self.output.fit(h, targets, solver=solver, **solver_kw)

            # Gradient step on encoders / biases
            pred = self.forward(x)
            loss = (pred - targets).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final decoder solve after last encoder update
        h = self._encode_hidden(x)
        self.output.fit(h, targets, solver=solver, **solver_kw)

    # ------------------------------------------------------------------
    # Strategy C — end-to-end SGD with NEF initialisation
    # ------------------------------------------------------------------

    def fit_end_to_end(self, x: Tensor, targets: Tensor,
                       n_epochs: int = 50, lr: float = 1e-3,
                       batch_size: int = 256,
                       loss: str = "mse") -> None:
        """Standard SGD on all parameters, initialised via greedy NEF solve.

        Args:
            loss: ``"mse"`` for regression / one-hot targets, or ``"ce"``
                  for cross-entropy (targets should be one-hot; converted
                  to class indices internally).
        """
        self.fit_greedy(x, targets)
        self.output.decoders.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs)

        if loss == "ce":
            loss_fn = nn.CrossEntropyLoss()
            train_targets = targets.argmax(dim=1)
        else:
            loss_fn = nn.MSELoss()
            train_targets = targets

        dataset = torch.utils.data.TensorDataset(x, train_targets)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

        for _ in range(n_epochs):
            for xb, yb in loader:
                pred = self.forward(xb)
                l = loss_fn(pred, yb)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            scheduler.step()

        self.output.decoders.requires_grad_(False)
