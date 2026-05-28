"""Test-wide reproducibility guards."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


MNIST_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def mnist_raw_present(root: Path | None = None) -> bool:
    """Return True only when all four raw MNIST IDX files are available."""
    root = root or Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "MNIST" / "raw"
    return all((raw_dir / name).is_file() for name in MNIST_RAW_FILES)


@pytest.fixture(autouse=True)
def _reset_global_rng_state():
    """Keep tests independent from ambient RNG state shared across the session."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
