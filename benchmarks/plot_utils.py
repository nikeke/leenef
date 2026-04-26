"""Matplotlib backend configuration and initialization for headless benchmark plotting."""

from __future__ import annotations

import matplotlib


def init_pyplot():
    """Force the non-interactive Agg backend and return pyplot."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
