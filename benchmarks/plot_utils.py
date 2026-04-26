"""Shared plotting helpers for benchmark scripts."""

from __future__ import annotations

import matplotlib


def init_pyplot():
    """Return pyplot after forcing the non-interactive Agg backend for headless scripts."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
