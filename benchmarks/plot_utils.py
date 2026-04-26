"""Shared plotting helpers for benchmark scripts."""

from __future__ import annotations

import matplotlib


def init_pyplot():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
