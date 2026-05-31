"""Preimage helpers for abs-activation NEF layers.

The helpers in this module expose the affine part of an ``NEFLayer`` with
``activation="abs"`` and solve the linear systems used in the abs-preimage
analysis scripts.  They deliberately return diagnostics for feasibility: a
least-squares preimage exists algebraically for every target, but it is only an
exact activity or one-hot preimage when the residual and sign-cell checks pass.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def affine_encoder(layer: Any) -> tuple[Tensor, Tensor]:
    """Return ``W, b`` for the affine pre-activation ``z = W x + b``.

    ``NEFLayer.encode`` computes ``abs(gain * (x @ encoders.T) + bias)`` for
    the default abs activation.  This helper exposes the row-stacked affine
    matrix ``W = diag(gain) @ encoders`` and the bias vector.
    """
    return layer.gain[:, None] * layer.encoders, layer.bias


def sign_cell(W: Tensor, b: Tensor, x: Tensor) -> Tensor:
    """Return ``sign(W x + b)`` with exact zeros assigned to ``+1``."""
    signs = torch.sign(W @ x + b)
    signs[signs == 0] = 1.0
    return signs


def activity_sign_preimage(
    W: Tensor, b: Tensor, target_activity: Tensor, signs: Tensor
) -> tuple[Tensor, dict[str, float]]:
    """Least-squares preimage for a target activity and fixed sign pattern.

    The returned input ``x`` solves ``W x ≈ signs * target_activity - b`` in the
    minimum-norm least-squares sense.  The diagnostics report whether this
    algebraic candidate is an exact activity preimage.
    """
    rhs = signs * target_activity - b
    x = torch.linalg.pinv(W) @ rhs
    preactivation = W @ x + b
    achieved = preactivation.abs()
    residual = W @ x - rhs
    diagnostics = {
        "linear_residual_norm": residual.norm().item(),
        "relative_activity_error": (
            (achieved - target_activity).norm() / target_activity.norm().clamp_min(1e-12)
        ).item(),
        "sign_flip_fraction": (sign_cell(W, b, x) != signs).float().mean().item(),
    }
    return x, diagnostics


def onehot_preimages_from_anchors(
    W: Tensor,
    b: Tensor,
    decoders: Tensor,
    anchors: Tensor,
    *,
    targets: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Solve class-target preimages anchored at representative inputs.

    For each anchor ``x_c``, the helper fixes the local abs sign cell
    ``s = sign(W x_c + b)`` and solves the linearized decode equation
    ``D.T diag(s) W x = target_c - D.T (s * b)`` for the closest point to the
    anchor.  The result is exact only for rows that remain in the same sign cell;
    diagnostics report sign consistency and true nonlinear decode deviation.
    """
    n_classes = decoders.shape[1]
    if targets is None:
        targets = torch.eye(n_classes, device=anchors.device, dtype=anchors.dtype)
    if targets.shape != (anchors.shape[0], n_classes):
        raise ValueError(
            f"targets must have shape ({anchors.shape[0]}, {n_classes}), "
            f"got {tuple(targets.shape)}"
        )

    xs = []
    in_cell = 0
    max_linear_residual = 0.0
    max_decode_dev = 0.0
    max_sign_flip_fraction = 0.0
    for anchor, target in zip(anchors, targets, strict=True):
        signs = sign_cell(W, b, anchor)
        M = (decoders.T * signs) @ W
        rhs = target - (decoders.T * signs) @ b
        x = anchor + torch.linalg.pinv(M) @ (rhs - M @ anchor)
        xs.append(x)

        new_signs = sign_cell(W, b, x)
        sign_flip_fraction = (new_signs != signs).float().mean().item()
        in_cell += int(sign_flip_fraction == 0.0)
        max_sign_flip_fraction = max(max_sign_flip_fraction, sign_flip_fraction)
        max_linear_residual = max(max_linear_residual, (M @ x - rhs).abs().max().item())
        logits = (W @ x + b).abs() @ decoders
        max_decode_dev = max(max_decode_dev, (logits - target).abs().max().item())

    return torch.stack(xs), {
        "in_cell_count": float(in_cell),
        "max_linear_residual": max_linear_residual,
        "max_decode_dev": max_decode_dev,
        "max_sign_flip_fraction": max_sign_flip_fraction,
    }


def nullspace_basis(matrix: Tensor, *, rtol: float | None = None) -> Tensor:
    """Return a row basis for the numerical right null space of ``matrix``."""
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    if rtol is None:
        rtol = torch.finfo(matrix.dtype).eps * max(matrix.shape)
    tol = singular_values.max().item() * rtol if singular_values.numel() else 0.0
    rank = int((singular_values > tol).sum().item())
    return vh[rank:]
