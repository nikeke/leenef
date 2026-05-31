"""Shared helpers for the abs-activation preimage explorations.

The NEF encode path is affine-then-abs::

    z = W x + b,   a = |z|,   W = diag(gain) @ encoders   (n_neurons x d_in)

so a single NEF layer is piecewise-linear: inside a sign cell
``s = sign(W x + b)`` the map ``x -> a`` is the linear map ``x -> s ⊙ (W x + b)``.
These helpers build a fitted layer on MNIST and expose ``W`` / ``b`` directly so
the exploration scripts can reason about the geometry analytically.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "benchmarks" / "sl_clean"))
sys.path.insert(0, str(_REPO / "src"))

from datasets import load_mnist  # noqa: E402

from leenef.layers import NEFLayer  # noqa: E402
from leenef.preimages import affine_encoder  # noqa: E402
from leenef.preimages import sign_cell as _sign_cell  # noqa: E402

FIG_DIR = _REPO / "docs" / "figures" / "abs_preimages"
FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR = _REPO / "results" / "abs_preimages"
RESULT_DIR.mkdir(exist_ok=True)

GENERATED_PATHS = (
    "docs/figures/abs_preimages",
    "docs/abs_preimage_geometry.pdf",
    "results/abs_preimages",
    "benchmarks/abs_preimages/prototype_initialization_result.json",
)


def _data_root() -> Path | None:
    """Resolve data roots for ordinary checkouts and linked worktrees."""
    env_root = os.environ.get("LEENEF_DATA_ROOT")
    if env_root:
        return Path(env_root)
    local_data = _REPO / "data"
    if (local_data / "MNIST" / "raw").exists():
        return local_data
    sibling_data = _REPO.parent / "leenef" / "data"
    if (sibling_data / "MNIST" / "raw").exists():
        return sibling_data
    return None


def load_mnist_subset(
    n_train: int = 10000, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load MNIST and subsample the training split for speed."""
    bundle = load_mnist(root=_data_root())
    xtr, ytr = bundle.x_train, bundle.y_train
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(xtr), generator=gen)[:n_train]
    return xtr[idx], ytr[idx], bundle.x_test, bundle.y_test


def _hash_tensor(tensor: Tensor) -> str:
    data = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(data.dtype).encode())
    h.update(str(tuple(data.shape)).encode())
    h.update(data.numpy().tobytes())
    return h.hexdigest()


def load_mnist_subset_with_metadata(
    n_train: int = 10000, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    """Load the standard subset and return hash metadata for provenance."""
    bundle = load_mnist(root=_data_root())
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(bundle.x_train), generator=gen)[:n_train]
    xtr = bundle.x_train[idx]
    ytr = bundle.y_train[idx]
    metadata = bundle.metadata()
    metadata["subset"] = {
        "n_train": int(n_train),
        "seed": int(seed),
        "indices_hash": _hash_tensor(idx),
        "x_train_subset": _hash_tensor(xtr),
        "y_train_subset": _hash_tensor(ytr),
        "x_test": _hash_tensor(bundle.x_test),
        "y_test": _hash_tensor(bundle.y_test),
    }
    root = _data_root()
    metadata["data_root"] = str(root) if root is not None else None
    return xtr, ytr, bundle.x_test, bundle.y_test, metadata


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=_REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def code_identity() -> dict[str, Any]:
    """Return HEAD plus a source-dirty hash that excludes generated outputs."""
    head = _run_git(["rev-parse", "HEAD"]).strip()
    pathspec = [".", *(f":(exclude){path}" for path in GENERATED_PATHS)]
    status = _run_git(["status", "--porcelain", "--", *pathspec])
    diff = _run_git(["diff", "--binary", "--", *pathspec])
    source_dirty = bool(status.strip() or diff)
    dirty_hash = None
    if source_dirty:
        h = hashlib.sha256()
        h.update(status.encode())
        h.update(diff.encode())
        dirty_hash = h.hexdigest()
    return {
        "head": head,
        "source_dirty": source_dirty,
        "source_dirty_hash": dirty_hash,
        "identity": f"{head}-dirty-{dirty_hash[:12]}" if dirty_hash else head,
        "excluded_generated_paths": list(GENERATED_PATHS),
    }


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "env_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, Path):
        return str(value.relative_to(_REPO) if value.is_relative_to(_REPO) else value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def image_stats(images: Tensor) -> dict[str, float]:
    data = images.detach().float().cpu()
    return {
        "min": data.min().item(),
        "max": data.max().item(),
        "mean": data.mean().item(),
        "std": data.std(unbiased=False).item(),
        "frac_lt_0": (data < 0).float().mean().item(),
        "frac_gt_1": (data > 1).float().mean().item(),
        "frac_outside_0_1": ((data < 0) | (data > 1)).float().mean().item(),
        "l2_norm_mean": data.reshape(data.shape[0], -1).norm(dim=1).mean().item()
        if data.ndim > 1
        else data.norm().item(),
    }


def write_result_json(
    filename: str,
    *,
    script: str,
    config: dict[str, Any],
    data_metadata: dict[str, Any],
    diagnostics: dict[str, Any],
    plotted_arrays: dict[str, Any] | None = None,
    artifact_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    payload = {
        "script": script,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "code_identity": code_identity(),
        "environment": environment_metadata(),
        "data_metadata": data_metadata,
        "config": config,
        "diagnostics": diagnostics,
        "plotted_arrays": plotted_arrays or {},
        "artifact_paths": [to_jsonable(path) for path in artifact_paths or []],
    }
    serializable = to_jsonable(payload)
    path = RESULT_DIR / filename
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(_REPO)}")
    return serializable


def build_layer(
    n_neurons: int, xtr: Tensor, ytr: Tensor, *, seed: int = 0
) -> tuple[NEFLayer, Tensor, Tensor]:
    """Fit a default-config NEF layer (abs, hypersphere, data-driven bias).

    Returns the layer plus the affine encoder pieces ``W = diag(gain) @ E``
    and ``b = bias`` so callers can analyse the piecewise-linear geometry.
    """
    rng = torch.Generator().manual_seed(seed)
    layer = NEFLayer(
        d_in=xtr.shape[1],
        n_neurons=n_neurons,
        d_out=10,
        activation="abs",
        gain=(0.5, 2.0),
        rng=rng,
        centers=xtr,
    )
    onehot = torch.zeros(len(ytr), 10)
    onehot[torch.arange(len(ytr)), ytr] = 1.0
    layer.fit(xtr, onehot)
    W, b = affine_encoder(layer)
    return layer, W, b


def sign_cell(W: Tensor, b: Tensor, x: Tensor) -> Tensor:
    """Re-export the abs-cell sign helper for the benchmark scripts."""
    return _sign_cell(W, b, x)


def decode(layer: NEFLayer, x: Tensor) -> tuple[Tensor, int]:
    """Softmax probabilities and argmax class for a single input."""
    logits = layer.encode(x.reshape(1, -1)).squeeze(0) @ layer.decoders
    return torch.softmax(logits, 0), int(logits.argmax())


def show(ax, img: Tensor, title: str, cmap: str = "gray") -> None:
    ax.imshow(img.reshape(28, 28).clamp(0, 1), cmap=cmap)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def quiet_proxychains() -> None:
    """No-op kept for symmetry; proxychains banners come from the shell."""
    os.environ.setdefault("MPLBACKEND", "Agg")
