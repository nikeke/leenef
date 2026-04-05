"""NEFEnsemble — ensemble of NEFLayers with different random seeds."""

import torch
import torch.nn as nn
from torch import Tensor

from .layers import NEFLayer


class NEFEnsemble(nn.Module):
    """Ensemble of independent :class:`NEFLayer` members.

    Each member is created with a different random seed, producing diverse
    random encoders.  Predictions are combined by averaging output
    probabilities (``combine="mean"``) or majority voting
    (``combine="vote"``).

    At 2 seconds per member on CPU, an ensemble of 20 trains in ~40 seconds
    — still faster than a single gradient-trained MLP — while substantially
    boosting accuracy through diversity of random projections.

    Args:
        n_members: number of ensemble members (default 10).
        base_seed: starting random seed; member *i* uses ``base_seed + i``.
        combine: aggregation strategy — ``"mean"`` averages outputs,
                 ``"vote"`` takes the most frequent argmax class.
        **layer_kwargs: all remaining arguments forwarded to each
                        :class:`NEFLayer` constructor (``d_in``, ``n_neurons``,
                        ``d_out``, ``activation``, ``encoder_strategy``,
                        ``gain``, ``centers``, etc.).
    """

    def __init__(
        self,
        *,
        n_members: int = 10,
        base_seed: int = 0,
        combine: str = "mean",
        **layer_kwargs,
    ):
        super().__init__()
        if combine not in ("mean", "vote"):
            raise ValueError(f"combine must be 'mean' or 'vote', got {combine!r}")
        self.combine = combine
        self.n_members = n_members

        members = []
        for i in range(n_members):
            rng = torch.Generator().manual_seed(base_seed + i)
            members.append(NEFLayer(**layer_kwargs, rng=rng))
        self.members = nn.ModuleList(members)

    @torch.no_grad()
    def fit(self, x: Tensor, targets: Tensor, solver: str = "tikhonov", **solver_kwargs) -> None:
        """Train all ensemble members independently.

        Each member encodes with its own random encoders and solves
        decoders analytically — no gradient computation.
        """
        for member in self.members:
            member.fit(x, targets, solver=solver, **solver_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Combine predictions from all ensemble members.

        Returns:
            If ``combine="mean"``: averaged outputs ``(N, d_out)``.
            If ``combine="vote"``: one-hot vote counts ``(N, d_out)``
            (argmax gives the majority class).
        """
        if self.combine == "mean":
            total = self.members[0](x)
            for member in self.members[1:]:
                total = total + member(x)
            return total / self.n_members

        # Majority vote: count how many members predict each class
        preds = torch.stack([m(x).argmax(dim=1) for m in self.members])  # (n_members, N)
        d_out = self.members[0].d_out
        votes = torch.zeros(x.shape[0], d_out, device=x.device)
        for i in range(self.n_members):
            votes.scatter_add_(
                1, preds[i].unsqueeze(1), torch.ones(x.shape[0], 1, device=x.device)
            )
        return votes
