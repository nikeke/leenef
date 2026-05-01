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
    """A preprocessed, hash-stamped dataset.

    ``x_train``/``x_test`` are 2D ``(N, d_in)`` for static datasets and
    3D ``(N, T, d_timestep)`` for sequential datasets.  ``sequence_shape``
    is ``(T, d_timestep)`` when sequential, ``None`` otherwise.
    """

    name: str
    x_train: Tensor  # (N, d_in) or (N, T, d) float32
    y_train: Tensor  # (N,) int64
    x_test: Tensor  # (M, d_in) or (M, T, d) float32
    y_test: Tensor  # (M,) int64
    n_classes: int
    preprocessing: str
    raw_hashes: dict
    processed_hashes: dict
    sequence_shape: tuple[int, int] | None = None

    def metadata(self) -> dict:
        meta = {
            "dataset": self.name,
            "n_train": int(self.x_train.shape[0]),
            "n_test": int(self.x_test.shape[0]),
            "n_classes": self.n_classes,
            "preprocessing": self.preprocessing,
            "raw_hashes": self.raw_hashes,
            "processed_hashes": self.processed_hashes,
        }
        if self.sequence_shape is None:
            meta["d_in"] = int(self.x_train.shape[1])
        else:
            T, d = self.sequence_shape
            meta["sequence_shape"] = [int(T), int(d)]
            meta["T"] = int(T)
            meta["d_timestep"] = int(d)
        return meta

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
    return _load_idx_dataset(name="MNIST", subdir="MNIST", preprocessing=preprocessing, root=root)


@lru_cache(maxsize=4)
def load_fashion_mnist(
    preprocessing: str = "div255_flat", root: Path | None = None
) -> DatasetBundle:
    """Load Fashion-MNIST as float32 tensors with deterministic preprocessing.

    Same on-disk format and shape as MNIST (28×28 grayscale, 10 classes).
    """
    return _load_idx_dataset(
        name="FashionMNIST",
        subdir="FashionMNIST",
        preprocessing=preprocessing,
        root=root,
    )


def _load_idx_dataset(
    *, name: str, subdir: str, preprocessing: str, root: Path | None
) -> DatasetBundle:
    """Shared IDX loader for MNIST-like datasets (28×28 grayscale, 10 classes)."""
    root = root or DATA_ROOT
    raw_dir = root / subdir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{name} raw files not found at {raw_dir}. Place the four IDX files under {raw_dir}/."
        )

    train_imgs, train_imgs_hash = _read_idx_images(raw_dir / "train-images-idx3-ubyte")
    train_lbls, train_lbls_hash = _read_idx_labels(raw_dir / "train-labels-idx1-ubyte")
    test_imgs, test_imgs_hash = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte")
    test_lbls, test_lbls_hash = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")

    if preprocessing == "div255_flat":
        x_train = _preprocess_unit_interval(train_imgs)
        x_test = _preprocess_unit_interval(test_imgs)
    elif preprocessing == "div255_flat_dc_shift":
        # MNIST decentered: shift by +0.5 then clamp to [0, 1].  Used in
        # the B1 follow-up "decentered MNIST" experiment to test the
        # DC-fold mechanism (Paper I §5.1, H7).
        x_train = (_preprocess_unit_interval(train_imgs) + 0.5).clamp_(0.0, 1.0)
        x_test = (_preprocess_unit_interval(test_imgs) + 0.5).clamp_(0.0, 1.0)
    else:
        raise ValueError(f"unknown preprocessing {preprocessing!r}")

    return DatasetBundle(
        name=name,
        x_train=x_train,
        y_train=train_lbls,
        x_test=x_test,
        y_test=test_lbls,
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
            "y_train": _hash_tensor(train_lbls),
            "x_test": _hash_tensor(x_test),
            "y_test": _hash_tensor(test_lbls),
        },
    )


@lru_cache(maxsize=2)
def load_cifar10(preprocessing: str = "div255_flat", root: Path | None = None) -> DatasetBundle:
    """Load CIFAR-10 as float32 tensors with deterministic preprocessing.

    Reads the standard ``cifar-10-batches-py`` Python-pickle layout.
    Each pickle file ships with its own MD5 in the upstream tarball; we
    additionally hash the raw bytes (SHA-256) so any local mutation is
    caught.

    Output shape: (N, 3072) float32, byte order RGBRGB...  (channel-major
    per CIFAR convention: first 1024 are red, next 1024 green, last 1024
    blue), normalized to [0, 1] via division by 255.
    """
    import pickle

    root = root or DATA_ROOT
    raw_dir = root / "cifar-10-batches-py"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"CIFAR-10 not found at {raw_dir}. Expected the standard cifar-10-batches-py layout."
        )

    train_files = [raw_dir / f"data_batch_{i}" for i in range(1, 6)]
    test_file = raw_dir / "test_batch"

    raw_hashes: dict[str, str] = {}
    train_x_list: list[Tensor] = []
    train_y_list: list[Tensor] = []
    for p in train_files:
        raw_bytes = p.read_bytes()
        raw_hashes[p.name] = hashlib.sha256(raw_bytes).hexdigest()
        d = pickle.loads(raw_bytes, encoding="bytes")
        train_x_list.append(torch.from_numpy(d[b"data"]).to(torch.uint8))
        train_y_list.append(torch.tensor(d[b"labels"], dtype=torch.int64))

    raw_bytes = test_file.read_bytes()
    raw_hashes[test_file.name] = hashlib.sha256(raw_bytes).hexdigest()
    d = pickle.loads(raw_bytes, encoding="bytes")
    test_x = torch.from_numpy(d[b"data"]).to(torch.uint8)
    test_y = torch.tensor(d[b"labels"], dtype=torch.int64)

    train_x = torch.cat(train_x_list, dim=0)  # (50000, 3072) uint8
    train_y = torch.cat(train_y_list, dim=0)  # (50000,) int64

    if preprocessing == "div255_flat":
        x_train = (train_x.to(torch.float32) / 255.0).contiguous()
        x_test = (test_x.to(torch.float32) / 255.0).contiguous()
    elif preprocessing == "div255_flat_per_channel_centered":
        # CIFAR-10 per-channel centred: subtract the per-channel mean
        # of the *training* set from both train and test.  Used in the
        # B1 follow-up "centred CIFAR-10" experiment to test the DC-fold
        # mechanism (Paper I §5.1, H7).  CIFAR raw layout is channel-major
        # within each row: 1024 R, 1024 G, 1024 B.
        x_train_raw = (train_x.to(torch.float32) / 255.0).contiguous()
        x_test_raw = (test_x.to(torch.float32) / 255.0).contiguous()
        means = torch.empty(3, dtype=torch.float32)
        for c in range(3):
            means[c] = x_train_raw[:, c * 1024 : (c + 1) * 1024].mean()
        ch_mean = torch.zeros(3072, dtype=torch.float32)
        for c in range(3):
            ch_mean[c * 1024 : (c + 1) * 1024] = means[c]
        x_train = (x_train_raw - ch_mean).contiguous()
        x_test = (x_test_raw - ch_mean).contiguous()
    else:
        raise ValueError(f"unknown preprocessing {preprocessing!r}")

    return DatasetBundle(
        name="CIFAR10",
        x_train=x_train,
        y_train=train_y,
        x_test=x_test,
        y_test=test_y,
        n_classes=10,
        preprocessing=preprocessing,
        raw_hashes=raw_hashes,
        processed_hashes={
            "x_train": _hash_tensor(x_train),
            "y_train": _hash_tensor(train_y),
            "x_test": _hash_tensor(x_test),
            "y_test": _hash_tensor(test_y),
        },
    )


_DATASET_LOADERS = {
    "MNIST": load_mnist,
    "FashionMNIST": load_fashion_mnist,
    "CIFAR10": load_cifar10,
}


# ---------------------------------------------------------------------------
# Sequential MNIST variants for the streaming/recurrent experiments.
#
# We reuse the MNIST raw bytes and just reshape.  The "permuted" variant
# applies a fixed permutation derived from a seeded ``torch.Generator`` so
# the dataset hash is stable across machines.  The permutation seed lives
# in the preprocessing string (``"smnist_permuted_seed=0"``) so downstream
# cache keys reflect the choice.
# ---------------------------------------------------------------------------


def _mnist_image_tensors(
    root: Path | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, str]]:
    """Return raw MNIST images/labels (uint8) and their raw hashes."""
    root = root or DATA_ROOT
    raw_dir = root / "MNIST" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"MNIST raw files not found at {raw_dir}. Place the four IDX files under {raw_dir}/."
        )
    train_imgs, ti_h = _read_idx_images(raw_dir / "train-images-idx3-ubyte")
    train_lbls, tl_h = _read_idx_labels(raw_dir / "train-labels-idx1-ubyte")
    test_imgs, ki_h = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte")
    test_lbls, kl_h = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")
    raw_hashes = {
        "train_images": ti_h,
        "train_labels": tl_h,
        "test_images": ki_h,
        "test_labels": kl_h,
    }
    return train_imgs, train_lbls, test_imgs, test_lbls, raw_hashes


@lru_cache(maxsize=2)
def load_smnist_row(root: Path | None = None) -> DatasetBundle:
    """Sequential MNIST in row-major form: ``(N, T=28, d=28)`` float32 in [0, 1].

    Each timestep is one row of the image; sequences are 28 timesteps long.
    """
    train_imgs, train_lbls, test_imgs, test_lbls, raw_hashes = _mnist_image_tensors(root)
    x_train = (train_imgs.to(torch.float32) / 255.0).contiguous()  # (N, 28, 28)
    x_test = (test_imgs.to(torch.float32) / 255.0).contiguous()
    return DatasetBundle(
        name="sMNIST-row",
        x_train=x_train,
        y_train=train_lbls,
        x_test=x_test,
        y_test=test_lbls,
        n_classes=10,
        preprocessing="smnist_row_div255",
        raw_hashes=raw_hashes,
        processed_hashes={
            "x_train": _hash_tensor(x_train),
            "y_train": _hash_tensor(train_lbls),
            "x_test": _hash_tensor(x_test),
            "y_test": _hash_tensor(test_lbls),
        },
        sequence_shape=(28, 28),
    )


@lru_cache(maxsize=4)
def load_smnist_permuted(perm_seed: int = 0, root: Path | None = None) -> DatasetBundle:
    """Permuted Sequential MNIST: ``(N, T=784, d=1)`` with a fixed pixel permutation.

    The permutation is derived from a seeded ``torch.Generator`` so the
    dataset hash is reproducible across machines.  ``perm_seed`` is part
    of the preprocessing string and the cache key.
    """
    train_imgs, train_lbls, test_imgs, test_lbls, raw_hashes = _mnist_image_tensors(root)
    g = torch.Generator().manual_seed(int(perm_seed))
    perm = torch.randperm(784, generator=g)
    x_train = (train_imgs.to(torch.float32) / 255.0).reshape(-1, 784)[:, perm]
    x_test = (test_imgs.to(torch.float32) / 255.0).reshape(-1, 784)[:, perm]
    x_train = x_train.unsqueeze(-1).contiguous()  # (N, 784, 1)
    x_test = x_test.unsqueeze(-1).contiguous()
    return DatasetBundle(
        name="sMNIST-permuted",
        x_train=x_train,
        y_train=train_lbls,
        x_test=x_test,
        y_test=test_lbls,
        n_classes=10,
        preprocessing=f"smnist_permuted_div255_seed={int(perm_seed)}",
        raw_hashes=raw_hashes,
        processed_hashes={
            "x_train": _hash_tensor(x_train),
            "y_train": _hash_tensor(train_lbls),
            "x_test": _hash_tensor(x_test),
            "y_test": _hash_tensor(test_lbls),
            "permutation": hashlib.sha256(perm.numpy().tobytes()).hexdigest(),
        },
        sequence_shape=(784, 1),
    )


_DATASET_LOADERS["sMNIST-row"] = lambda preprocessing="smnist_row_div255": load_smnist_row()
_DATASET_LOADERS["sMNIST-permuted-seed0"] = lambda preprocessing="smnist_permuted_div255_seed=0": (
    load_smnist_permuted(perm_seed=0)
)


def load_dataset(name: str, preprocessing: str = "div255_flat") -> DatasetBundle:
    """Dispatch by name. Adds a single point of entry for the experiments
    module so a new dataset only requires a loader registration."""
    if name not in _DATASET_LOADERS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(_DATASET_LOADERS)}")
    return _DATASET_LOADERS[name](preprocessing=preprocessing)
