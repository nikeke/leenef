"""Utilities for deterministic substream derivation."""

from __future__ import annotations

import torch

_MAX_SEED = 2**31 - 1


def fork_generator(
    rng: torch.Generator | None,
    *,
    fallback_seed: int = 0,
    device: str | torch.device = "cpu",
) -> torch.Generator:
    """Create a deterministic child generator from ``rng``.

    When a parent generator is provided, the child seed is sampled from that
    parent so downstream randomness is explicit and reproducible without
    depending on ambient global RNG state. When no parent is available, the
    fallback seed keeps the caller deterministic by default.
    """

    child = torch.Generator(device=device)
    if rng is None:
        child.manual_seed(fallback_seed)
        return child
    parent_device = getattr(rng, "device", torch.device("cpu"))
    seed = int(torch.randint(0, _MAX_SEED, (1,), generator=rng, device=parent_device).item())
    child.manual_seed(seed)
    return child
