"""Continual-learning datasets and task-stream constructors.

Re-exports the deterministic raw loaders from :mod:`benchmarks.sl_clean.datasets`
(MNIST, FashionMNIST, CIFAR-10, sMNIST variants) so cl_clean does not
duplicate IDX/CIFAR parsing, then layers CL-specific helpers on top:

* :func:`load_cifar100` — pickle-format CIFAR-100 with the same hash
  discipline as :func:`benchmarks.sl_clean.datasets.load_cifar10`.
* :func:`make_task_stream` — split a static dataset bundle into a
  reproducible sequence of tasks under one of two scenarios:

  - ``"split"`` (class-incremental): partition the label set into
    ``n_tasks`` disjoint groups; each task contains only the examples
    whose labels fall in its group.
  - ``"permuted"`` (domain-incremental): use the full dataset every
    task, but apply a fresh fixed pixel permutation per task.  Labels
    are unchanged.

The task ordering — the random class assignment and the per-task
permutation matrices — is derived deterministically from
``(stream_seed, scenario, n_tasks)`` via an explicit
:class:`torch.Generator`.  Each :class:`TaskStream` carries a
``stream_hash`` that hashes the concrete realization (class
assignments and/or permutation tensors), so a cl_clean result row
records exactly which task stream it consumed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from benchmarks.sl_clean.datasets import (
    DATA_ROOT,
    DatasetBundle,
    _hash_tensor,
    load_cifar10,
    load_fashion_mnist,
    load_mnist,
    load_smnist_permuted,
    load_smnist_row,
)

# ---------------------------------------------------------------------------
# CIFAR-100 — pickled in the upstream "cifar-100-python" layout.
# ---------------------------------------------------------------------------


def load_cifar100(preprocessing: str = "div255_flat", root: Path | None = None) -> DatasetBundle:
    """Load CIFAR-100 as float32 tensors with deterministic preprocessing.

    Reads the standard ``cifar-100-python`` layout (``train`` and ``test``
    pickle files).  Output shape: ``(N, 3072)`` float32, RGB channel-major
    ordering matching :func:`load_cifar10`.
    """
    import pickle

    root = root or DATA_ROOT
    raw_dir = root / "cifar-100-python"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"CIFAR-100 not found at {raw_dir}. Expected the standard cifar-100-python layout."
        )

    raw_hashes: dict[str, str] = {}

    def _read(path: Path) -> tuple[Tensor, Tensor]:
        raw = path.read_bytes()
        raw_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        d = pickle.loads(raw, encoding="bytes")
        x = torch.from_numpy(d[b"data"]).to(torch.uint8)
        y = torch.tensor(d[b"fine_labels"], dtype=torch.int64)
        return x, y

    train_x, train_y = _read(raw_dir / "train")
    test_x, test_y = _read(raw_dir / "test")

    if preprocessing == "div255_flat":
        x_train = (train_x.to(torch.float32) / 255.0).contiguous()
        x_test = (test_x.to(torch.float32) / 255.0).contiguous()
    else:
        raise ValueError(f"unknown preprocessing {preprocessing!r}")

    return DatasetBundle(
        name="CIFAR100",
        x_train=x_train,
        y_train=train_y,
        x_test=x_test,
        y_test=test_y,
        n_classes=100,
        preprocessing=preprocessing,
        raw_hashes=raw_hashes,
        processed_hashes={
            "x_train": _hash_tensor(x_train),
            "y_train": _hash_tensor(train_y),
            "x_test": _hash_tensor(x_test),
            "y_test": _hash_tensor(test_y),
        },
    )


_BASE_LOADERS = {
    "MNIST": lambda: load_mnist(),
    "FashionMNIST": lambda: load_fashion_mnist(),
    "CIFAR10": lambda: load_cifar10(),
    "CIFAR100": lambda: load_cifar100(),
    "sMNIST-row": lambda: load_smnist_row(),
    "sMNIST-permuted-seed0": lambda: load_smnist_permuted(perm_seed=0),
}


def load_base_dataset(name: str) -> DatasetBundle:
    """Dispatch by name to the correct base-dataset loader."""
    if name not in _BASE_LOADERS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(_BASE_LOADERS)}")
    return _BASE_LOADERS[name]()


# ---------------------------------------------------------------------------
# Task streams.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSplit:
    """One task in a continual-learning stream."""

    task_index: int
    classes: tuple[int, ...]
    x_train: Tensor
    y_train: Tensor
    x_test: Tensor
    y_test: Tensor
    permutation: Tensor | None = None

    def train_inputs(
        self,
        *,
        start: int | None = None,
        end: int | None = None,
        indices: Tensor | None = None,
    ) -> Tensor:
        x = self.x_train
        if indices is not None:
            x = x[indices]
        elif start is not None or end is not None:
            x = x[slice(start, end)]
        if self.permutation is None:
            return x
        return x[:, self.permutation].contiguous()

    def test_inputs(self, *, start: int | None = None, end: int | None = None) -> Tensor:
        x = self.x_test if start is None and end is None else self.x_test[slice(start, end)]
        if self.permutation is None:
            return x
        return x[:, self.permutation].contiguous()


@dataclass(frozen=True)
class TaskStream:
    """A reproducible CL task stream with a hashable realization."""

    base_name: str
    scenario: str  # "split" or "permuted"
    n_tasks: int
    n_classes: int
    stream_seed: int
    tasks: tuple[TaskSplit, ...]
    stream_hash: str
    raw_hashes: dict
    processed_hashes: dict
    extra: dict

    def metadata(self) -> dict:
        return {
            "base_dataset": self.base_name,
            "scenario": self.scenario,
            "n_tasks": int(self.n_tasks),
            "n_classes": int(self.n_classes),
            "stream_seed": int(self.stream_seed),
            "stream_hash": self.stream_hash,
            "raw_hashes": dict(self.raw_hashes),
            "processed_hashes": dict(self.processed_hashes),
            "task_classes": [list(t.classes) for t in self.tasks],
            **self.extra,
        }


def _split_stream(bundle: DatasetBundle, *, n_tasks: int, stream_seed: int) -> TaskStream:
    """Class-incremental partition of ``bundle`` into ``n_tasks`` disjoint groups."""
    n_classes = bundle.n_classes
    if n_classes % n_tasks != 0:
        raise ValueError(
            f"split stream: n_classes={n_classes} must be divisible by n_tasks={n_tasks}"
        )
    per_task = n_classes // n_tasks

    g = torch.Generator().manual_seed(stream_seed * 2_654_435_761 + 1)
    class_perm = torch.randperm(n_classes, generator=g).tolist()

    tasks: list[TaskSplit] = []
    for t in range(n_tasks):
        cls = tuple(sorted(class_perm[t * per_task : (t + 1) * per_task]))
        train_mask = torch.zeros(bundle.y_train.shape[0], dtype=torch.bool)
        for c in cls:
            train_mask |= bundle.y_train == c
        test_mask = torch.zeros(bundle.y_test.shape[0], dtype=torch.bool)
        for c in cls:
            test_mask |= bundle.y_test == c
        tasks.append(
            TaskSplit(
                task_index=t,
                classes=cls,
                x_train=bundle.x_train[train_mask].contiguous(),
                y_train=bundle.y_train[train_mask].contiguous(),
                x_test=bundle.x_test[test_mask].contiguous(),
                y_test=bundle.y_test[test_mask].contiguous(),
            )
        )

    h = hashlib.sha256()
    h.update(b"split\0")
    h.update(int(n_tasks).to_bytes(8, "little", signed=False))
    h.update(int(n_classes).to_bytes(8, "little", signed=False))
    h.update(int(stream_seed).to_bytes(8, "little", signed=True))
    for c in class_perm:
        h.update(int(c).to_bytes(8, "little", signed=True))
    stream_hash = h.hexdigest()
    return TaskStream(
        base_name=bundle.name,
        scenario="split",
        n_tasks=n_tasks,
        n_classes=n_classes,
        stream_seed=stream_seed,
        tasks=tuple(tasks),
        stream_hash=stream_hash,
        raw_hashes=bundle.raw_hashes,
        processed_hashes=bundle.processed_hashes,
        extra={"class_permutation": class_perm},
    )


def _permuted_stream(bundle: DatasetBundle, *, n_tasks: int, stream_seed: int) -> TaskStream:
    """Domain-incremental: same labels, fresh pixel permutation per task.

    The task objects share the base train/test tensors and only carry
    their permutation. Inputs are materialized on demand via
    ``TaskSplit.train_inputs`` / ``test_inputs`` so large B3 grids do not
    keep one full copied dataset per task resident in memory.
    """
    if bundle.sequence_shape is not None:
        raise ValueError(
            f"permuted stream expects a flat dataset; got sequence_shape={bundle.sequence_shape}"
        )
    d_in = int(bundle.x_train.shape[1])

    g = torch.Generator().manual_seed(stream_seed * 2_654_435_761 + 7)
    perms = [torch.randperm(d_in, generator=g) for _ in range(n_tasks)]

    tasks: list[TaskSplit] = []
    all_classes = tuple(range(bundle.n_classes))
    for t, perm in enumerate(perms):
        tasks.append(
            TaskSplit(
                task_index=t,
                classes=all_classes,
                x_train=bundle.x_train,
                y_train=bundle.y_train,
                x_test=bundle.x_test,
                y_test=bundle.y_test,
                permutation=perm,
            )
        )

    perm_hash = hashlib.sha256(b"|".join(p.numpy().tobytes() for p in perms)).hexdigest()
    h = hashlib.sha256()
    h.update(b"permuted\0")
    h.update(int(n_tasks).to_bytes(8, "little", signed=False))
    h.update(int(bundle.n_classes).to_bytes(8, "little", signed=False))
    h.update(int(stream_seed).to_bytes(8, "little", signed=True))
    h.update(perm_hash.encode("ascii"))
    stream_hash = h.hexdigest()
    return TaskStream(
        base_name=bundle.name,
        scenario="permuted",
        n_tasks=n_tasks,
        n_classes=bundle.n_classes,
        stream_seed=stream_seed,
        tasks=tuple(tasks),
        stream_hash=stream_hash,
        raw_hashes=bundle.raw_hashes,
        processed_hashes={**bundle.processed_hashes, "permutations": perm_hash},
        extra={"d_in": d_in, "n_permutations": n_tasks},
    )


def make_task_stream(
    base_name: str, *, scenario: str, n_tasks: int, stream_seed: int
) -> TaskStream:
    """Construct a deterministic task stream from a base dataset.

    ``base_name`` is one of :data:`_BASE_LOADERS`.  ``scenario`` is
    ``"split"`` (class-incremental) or ``"permuted"`` (domain-incremental).
    The result is fully determined by ``(base_name, scenario, n_tasks,
    stream_seed)``: same arguments always produce the same task ordering
    and the same ``stream_hash``.
    """
    bundle = load_base_dataset(base_name)
    if scenario == "split":
        return _split_stream(bundle, n_tasks=n_tasks, stream_seed=stream_seed)
    if scenario == "permuted":
        return _permuted_stream(bundle, n_tasks=n_tasks, stream_seed=stream_seed)
    raise ValueError(f"unknown scenario {scenario!r}; expected 'split' or 'permuted'")
