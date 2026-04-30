"""Deterministic dataset loaders for ``sl_clean`` experiments.

We avoid ``torchvision.datasets`` *Datasets* and just read the raw IDX
files directly: it gives us full control over byte ordering, dtype,
preprocessing, and lets us hash the resulting tensors so every result
row records exactly which bytes it consumed.

Hash policy:

* ``raw_*_hash`` is a SHA-256 of the raw IDX bytes on disk.
* ``processed_*_hash`` is a SHA-256 of the ``contiguous().numpy().tobytes()``
  of the post-preprocessing float32 tensor (or int64 for labels).

Both are recorded in :func:`load_mnist`'s returned ``DatasetBundle`` so
they can be plumbed into the per-result metadata.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _read_idx_images(path: Path) -> tuple[Tensor, str]:
    """Read an IDX image file → (uint8 tensor of shape (N, H, W), raw_hash)."""
    raw = path.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    magic, n, h, w = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"{path}: bad IDX image magic {magic}")
    arr = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).reshape(n, h, w)
    return arr.clone(), raw_hash


def _read_idx_labels(path: Path) -> tuple[Tensor, str]:
    """Read an IDX label file → (int64 tensor of shape (N,), raw_hash)."""
    raw = path.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    magic, n = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"{path}: bad IDX label magic {magic}")
    arr = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).reshape(n).to(torch.int64)
    return arr, raw_hash


def _hash_tensor(t: Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class DatasetBundle:
    """A preprocessed, hash-stamped dataset."""

    name: str
    x_train: Tensor  # (N, d_in) float32
    y_train: Tensor  # (N,) int64
    x_test: Tensor  # (M, d_in) float32
    y_test: Tensor  # (M,) int64
    n_classes: int
    preprocessing: str
    raw_hashes: dict
    processed_hashes: dict

    def metadata(self) -> dict:
        return {
            "dataset": self.name,
            "n_train": int(self.x_train.shape[0]),
            "n_test": int(self.x_test.shape[0]),
            "d_in": int(self.x_train.shape[1]),
            "n_classes": self.n_classes,
            "preprocessing": self.preprocessing,
            "raw_hashes": self.raw_hashes,
            "processed_hashes": self.processed_hashes,
        }

    def one_hot(self, y: Tensor) -> Tensor:
        oh = torch.zeros(y.shape[0], self.n_classes, dtype=torch.float32)
        oh[torch.arange(y.shape[0]), y] = 1.0
        return oh


def _preprocess_unit_interval(x: Tensor) -> Tensor:
    """Flatten to (N, d) float32 in [0, 1]. Matches the legacy TR protocol."""
    return (x.to(torch.float32) / 255.0).reshape(x.shape[0], -1).contiguous()


@lru_cache(maxsize=4)
def load_mnist(preprocessing: str = "div255_flat", root: Path | None = None) -> DatasetBundle:
    """Load MNIST as float32 tensors with deterministic preprocessing.

    Cached so repeated calls within a process are free.

    Args:
        preprocessing: one of:
            ``"div255_flat"`` — divide by 255 and flatten (matches legacy TR).
        root: override data root (defaults to ``<repo>/data``).
    """
    root = root or DATA_ROOT
    raw_dir = root / "MNIST" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"MNIST raw files not found at {raw_dir}. Place the four IDX files under {raw_dir}/."
        )

    train_imgs, train_imgs_hash = _read_idx_images(raw_dir / "train-images-idx3-ubyte")
    train_lbls, train_lbls_hash = _read_idx_labels(raw_dir / "train-labels-idx1-ubyte")
    test_imgs, test_imgs_hash = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte")
    test_lbls, test_lbls_hash = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")

    if preprocessing == "div255_flat":
        x_train = _preprocess_unit_interval(train_imgs)
        x_test = _preprocess_unit_interval(test_imgs)
    else:
        raise ValueError(f"unknown preprocessing {preprocessing!r}")

    y_train = train_lbls
    y_test = test_lbls

    return DatasetBundle(
        name="MNIST",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_classes=10,
        preprocessing=preprocessing,
        raw_hashes={
            "train_images": train_imgs_hash,
            "train_labels": train_lbls_hash,
            "test_images": test_imgs_hash,
            "test_labels": test_lbls_hash,
        },
        processed_hashes={
            "x_train": _hash_tensor(x_train),
            "y_train": _hash_tensor(y_train),
            "x_test": _hash_tensor(x_test),
            "y_test": _hash_tensor(y_test),
        },
    )
