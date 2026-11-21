"""Neuron selection from sufficient statistics and data-directed proposals."""

import operator

import torch
from torch import Tensor

from .layers import NEFLayer
from .solvers import (
    DEFAULT_ALPHA,
    AlphaPolicy,
    downdate_inverse,
    inverse_from_normal_equations,
    regularization_strength,
)


def _valid_inverse_diagonal(inverse: Tensor) -> bool:
    diagonal = inverse.diagonal()
    return bool(((diagonal > 0) & diagonal.isfinite()).all())


def _saliency(inverse: Tensor, cross_matrix: Tensor) -> Tensor:
    decoders = inverse @ cross_matrix
    return decoders.square().sum(dim=1) / inverse.diagonal()


@torch.no_grad()
def obs_saliency(
    normal_matrix: Tensor, cross_matrix: Tensor, *, alpha: AlphaPolicy = DEFAULT_ALPHA
) -> Tensor:
    """Return float64 optimal-brain-surgeon scores for ``A.T @ A`` and ``A.T @ Y``.

    Each score is the exact increase in the minimum ridge objective after
    deleting that neuron and refitting at the same regularizer. A trace policy
    is evaluated once on the supplied normal matrix. Raises ``RuntimeError``
    if the inverse diagonal contains non-positive or non-finite entries.
    """
    inverse = inverse_from_normal_equations(normal_matrix.double(), alpha=alpha)
    if not _valid_inverse_diagonal(inverse):
        raise RuntimeError("OBS saliency requires a finite, positive inverse diagonal")
    return _saliency(inverse, cross_matrix.double())


@torch.no_grad()
def prune_from_stats(
    normal_matrix: Tensor,
    cross_matrix: Tensor,
    n_keep: int,
    *,
    alpha: AlphaPolicy = DEFAULT_ALPHA,
    drop_fraction: float = 0.1,
    method: str = "refit",
) -> Tensor:
    """Greedily delete the least salient neurons and return sorted original indices.

    Each round refreshes the OBS scores and drops at least one neuron, capped
    at the requested width. Ties are broken by dropping lower indices first.
    ``refit`` re-evaluates the alpha policy on each smaller normal matrix;
    ``downdate`` evaluates it on the initial matrix and holds it fixed.
    If a downdated inverse has an invalid diagonal, it is rebuilt at the
    same fixed regularizer before scoring the remaining neurons.
    """
    n = normal_matrix.shape[0]
    try:
        n_keep = operator.index(n_keep)
    except TypeError as exc:
        raise ValueError("n_keep must be an integer") from exc
    if not 1 <= n_keep <= n:
        raise ValueError(f"n_keep must be between 1 and {n}, got {n_keep}")
    if not 0 < drop_fraction < 1:
        raise ValueError(f"drop_fraction must be between 0 and 1, got {drop_fraction}")
    if method not in ("refit", "downdate"):
        raise ValueError(f"Unknown pruning method {method!r}; expected 'refit' or 'downdate'")

    keep = torch.arange(n, device=normal_matrix.device)
    if n_keep == n:
        return keep
    normal = normal_matrix.double()
    cross = cross_matrix.double()
    if method == "downdate":
        alpha = (float(regularization_strength(normal, alpha=alpha)), "fixed")
    inverse = inverse_from_normal_equations(normal, alpha=alpha)
    while n > n_keep:
        valid = _valid_inverse_diagonal(inverse)
        if not valid and method == "downdate" and n < normal.shape[0]:
            inverse = inverse_from_normal_equations(normal[keep[:, None], keep], alpha=alpha)
            valid = _valid_inverse_diagonal(inverse)
        if not valid:
            raise RuntimeError("OBS saliency requires a finite, positive inverse diagonal")
        n_drop = min(max(1, int(drop_fraction * n)), n - n_keep)
        order = torch.argsort(_saliency(inverse, cross), stable=True)
        survivors = order[n_drop:].sort().values
        keep = keep[survivors]
        n -= n_drop
        if n > n_keep:
            cross = cross[survivors]
            if method == "refit":
                normal = normal[survivors[:, None], survivors]
                inverse = inverse_from_normal_equations(normal, alpha=alpha)
            else:
                inverse = downdate_inverse(inverse, survivors, order[:n_drop])
    return keep


@torch.no_grad()
def residual_moments(
    x: Tensor,
    residual: Tensor,
    *,
    mean: Tensor | None = None,
    out: Tensor | None = None,
    batch_size: int = 8192,
) -> Tensor:
    """Accumulate residual-weighted second moments of centered inputs.

    ``x`` and ``residual`` have shapes ``(N, d_in)`` and ``(N, d_out)``.
    The result has shape ``(d_out, d_in, d_in)`` and dtype float64; each
    batch's matrix products use the input dtype. ``mean`` is a fixed center
    (zero if omitted). Pass ``out`` to add to an existing result in place.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    d_in, d_out = x.shape[1], residual.shape[1]
    shape = (d_out, d_in, d_in)
    if out is None:
        out = torch.zeros(shape, device=x.device, dtype=torch.float64)
    elif out.shape != shape or out.dtype != torch.float64 or out.device != x.device:
        raise ValueError(f"out must have shape {shape}, dtype float64 and be on x.device")
    if mean is not None:
        mean = mean.to(x)
    for start in range(0, x.shape[0], batch_size):
        xc = x[start : start + batch_size]
        if mean is not None:
            xc = xc - mean
        r = residual[start : start + batch_size].to(xc)
        for c in range(d_out):
            out[c].add_((xc * r[:, c : c + 1]).T @ xc)
    return out


@torch.no_grad()
def phd_bases(moments: Tensor, n_components: int) -> Tensor:
    """Return float32 pHd bases of shape ``(d_out, d_in, min(n_components, d_in))``.

    Each basis contains eigenvectors of the symmetrized residual moment,
    ordered by decreasing absolute eigenvalue and scaled by its square root.
    """
    if n_components < 0:
        raise ValueError("n_components must be non-negative")
    moments = moments.double()
    # This eigendecomposition orders residual curvature; it is not a ridge solve.
    values, vectors = torch.linalg.eigh((moments + moments.mT) * 0.5)
    magnitude = values.abs()
    order = magnitude.argsort(dim=-1, descending=True, stable=True)[..., :n_components]
    basis = vectors.gather(-1, order.unsqueeze(-2).expand(-1, vectors.shape[-2], -1))
    return (basis * magnitude.gather(-1, order).sqrt().unsqueeze(-2)).float()


def _randn(shape: tuple[int, ...], device: torch.device, rng: torch.Generator) -> Tensor:
    return torch.randn(shape, generator=rng, device=rng.device, dtype=torch.float32).to(device)


def _normalize_directions(directions: Tensor, rng: torch.Generator) -> Tensor:
    norms = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    zero_rows = (norms[:, 0] == 0).nonzero(as_tuple=True)[0]
    if zero_rows.numel() > 0:
        fallback = _randn((zero_rows.numel(), directions.shape[1]), directions.device, rng)
        directions[zero_rows] = fallback
        norms[zero_rows] = torch.linalg.vector_norm(fallback, dim=1, keepdim=True)
    return directions.div_(norms)


@torch.no_grad()
def sample_directions_from_bases(n_neurons: int, bases: Tensor, *, rng: torch.Generator) -> Tensor:
    """Sample unit-norm float32 directions from Gaussian combinations of pHd bases.

    Each neuron independently chooses an output's basis uniformly. Zero
    combinations fall back to isotropic Gaussian directions from ``rng``.
    """
    d_out, d_in, q = bases.shape
    bases = bases.float()
    cls = torch.randint(d_out, (n_neurons,), generator=rng, device=rng.device).to(bases.device)
    coeff = _randn((n_neurons, q), bases.device, rng)
    directions = bases.new_empty((n_neurons, d_in))
    # Bound the memory of the gathered per-neuron bases.
    chunk_size = max(1, (1 << 30) // max(1, d_in * q * bases.element_size()))
    for start in range(0, n_neurons, chunk_size):
        chunk = slice(start, start + chunk_size)
        directions[chunk] = torch.einsum("nq,ndq->nd", coeff[chunk], bases[cls[chunk]])
    return _normalize_directions(directions, rng)


@torch.no_grad()
def agop(layer: NEFLayer, x: Tensor, *, batch_size: int = 4096) -> Tensor:
    """Average the fitted layer's input-Jacobian outer products in float64.

    Uses the derivative of the layer's elementwise activation at each
    preactivation, taking zero for disconnected threshold activations.
    """
    if batch_size <= 0 or x.shape[0] == 0:
        raise ValueError("agop requires at least one sample and a positive batch_size")
    result = torch.zeros((layer.d_in, layer.d_in), device=x.device, dtype=torch.float64)
    encoders, gain, decoders = (
        layer.encoders.double(),
        layer.gain.double(),
        layer.decoders.double(),
    )
    for batch in x.split(batch_size):
        z = layer.gain * (batch @ layer.encoders.T) + layer.bias
        with torch.enable_grad():
            z.requires_grad_()
            rates = layer.activation(z)
            if not rates.requires_grad:
                continue
            derivative = torch.autograd.grad(rates.sum(), z, allow_unused=True)[0]
        if derivative is None:
            continue
        slopes = derivative.double() * gain
        for weights in decoders.T:
            jacobian = (slopes * weights) @ encoders
            result.addmm_(jacobian.T, jacobian)
    return result.div_(x.shape[0])


@torch.no_grad()
def psd_power(matrix: Tensor, power: float) -> Tensor:
    """Return the float32 power of a symmetric PSD matrix after spectral normalization.

    Negative eigenvalues are clamped to zero before dividing by the largest
    eigenvalue. A zero spectrum stays zero before exponentiation; negative
    powers require a strictly positive spectrum.
    """
    matrix = matrix.double()
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    values.clamp_min_(0)
    if power < 0 and (values == 0).any():
        raise ValueError("negative matrix powers require a strictly positive spectrum")
    largest = values[-1]
    values = values / torch.where(largest > 0, largest, 1.0)
    return ((vectors * values.pow(power)) @ vectors.T).float()


@torch.no_grad()
def sample_shaped_directions(n_neurons: int, shaping: Tensor, *, rng: torch.Generator) -> Tensor:
    """Shape Gaussian draws by right-multiplication and return unit float32 rows.

    Zero rows fall back to isotropic Gaussian directions from ``rng``.
    """
    draws = _randn((n_neurons, shaping.shape[0]), shaping.device, rng)
    return _normalize_directions(draws @ shaping.float(), rng)
