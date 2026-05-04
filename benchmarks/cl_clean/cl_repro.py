"""Shared reproducibility helpers for continual-learning clean sweeps.

This is a direct mirror of ``benchmarks.sl_clean.sl_repro`` with the
namespace bumped to ``cl_clean`` and a separate runner-version tag.
Two namespaces means CL result rows and SL result rows never collide
or get cross-validated.

The contract is the same:

* every result row stores enough metadata that the runner can decide
  whether it was produced by *the current code, current configuration,
  current run context, and current seed* — and skip it on resume if so;
* ``code_identity`` is a content hash over ``CODE_IDENTITY_PATHS``
  (``src``, ``benchmarks``, ``tests``, ``pyproject.toml``).  Generated
  artifacts are deliberately excluded so docs/results commits do not
  invalidate cached cells.

Per-experiment paths and run-context schemas live in the experiment
registry (``benchmarks.cl_clean.experiments``) — this module is
deliberately experiment-agnostic.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

RESULT_NAMESPACE = "cl_clean"
RUNNER_VERSION = "cl-clean-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / RESULT_NAMESPACE
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
DEFAULT_BASE_CONFIG_PATH = RESULTS_DIR / "base_config.json"
DOC_PATH = REPO_ROOT / "docs" / "cl_clean_sweeps.md"

CODE_IDENTITY_PATHS: tuple[str, ...] = (
    "src",
    "benchmarks",
    "tests",
    "pyproject.toml",
)


def set_cl_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch deterministically."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_policy() -> dict[str, str]:
    return {
        "main_seed": "per-run user-visible seed",
        "torch_seed": "torch.manual_seed(seed)",
        "numpy_seed": "np.random.seed(seed)",
        "python_seed": "random.seed(seed)",
        "task_stream_seed": "derived deterministically from seed inside the experiment",
    }


def _git(*args: str, repo_root: Path = REPO_ROOT) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return out.stdout


@lru_cache(maxsize=1)
def code_identity() -> str | None:
    """Identity for the source tree, ignoring generated artifacts."""
    head_out = _git("rev-parse", "HEAD")
    if not head_out:
        return None
    head = head_out.strip()
    if not head:
        return None
    tree_listing = _git("ls-tree", "-r", "HEAD", "--", *CODE_IDENTITY_PATHS) or ""
    content_id = hashlib.sha256(tree_listing.encode("utf-8")).hexdigest()[:40]
    diff = _git("diff", "--no-ext-diff", "HEAD", "--", *CODE_IDENTITY_PATHS) or ""
    untracked_list = (
        _git("ls-files", "--others", "--exclude-standard", "--", *CODE_IDENTITY_PATHS) or ""
    )
    untracked_files = [line for line in untracked_list.splitlines() if line.strip()]
    if not diff and not untracked_files:
        return content_id
    h = hashlib.sha256()
    h.update(diff.encode("utf-8"))
    h.update(b"\n--untracked--\n")
    for rel in sorted(untracked_files):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        path = REPO_ROOT / rel
        try:
            h.update(path.read_bytes())
        except (OSError, IsADirectoryError):
            h.update(b"<unreadable>")
        h.update(b"\n")
    dirty = h.hexdigest()[:12]
    return f"{content_id}-dirty-{dirty}"


def reset_code_identity_cache() -> None:
    code_identity.cache_clear()


def hardware_info() -> dict[str, object]:
    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["cuda_device_name"] = None
    return info


def result_metadata(
    *,
    config: dict[str, object],
    seed: int,
    experiment: str,
    run_context: dict[str, object] | None = None,
    primary_metric: str | None = None,
    higher_is_better: bool = True,
) -> dict[str, object]:
    return {
        "result_namespace": RESULT_NAMESPACE,
        "runner_version": RUNNER_VERSION,
        "experiment": experiment,
        "config": dict(config),
        "seed": int(seed),
        "seed_policy": seed_policy(),
        "code_identity": code_identity(),
        "run_context": dict(run_context or {}),
        "primary_metric": primary_metric,
        "higher_is_better": bool(higher_is_better),
        "hardware": hardware_info(),
    }


def can_reuse_result(
    result: dict[str, object],
    *,
    experiment: str,
    expected_config: dict[str, object],
    expected_seed: int,
    expected_run_context: dict[str, object] | None = None,
) -> bool:
    if "error" in result:
        return False
    return (
        result.get("result_namespace") == RESULT_NAMESPACE
        and result.get("runner_version") == RUNNER_VERSION
        and result.get("experiment") == experiment
        and result.get("config") == expected_config
        and result.get("seed") == expected_seed
        and result.get("seed_policy") == seed_policy()
        and result.get("code_identity") == code_identity()
        and result.get("run_context", {}) == (expected_run_context or {})
    )


def expected_config_with_knob(
    base_config: dict[str, object], knob: str, value: object
) -> dict[str, object]:
    if knob not in base_config:
        raise KeyError(f"knob {knob!r} not present in base_config (keys={sorted(base_config)})")
    cfg = dict(base_config)
    cfg[knob] = value
    return cfg


def is_current_sweep_row(
    result: dict[str, object],
    *,
    experiment: str,
    base_config: dict[str, object],
    knob: str,
    expected_run_context: dict[str, object],
) -> bool:
    if "error" in result:
        return False
    if result.get("knob") != knob or "value" not in result:
        return False
    seed = result.get("seed")
    if not isinstance(seed, int):
        return False
    expected_config = expected_config_with_knob(base_config, knob, result["value"])
    return can_reuse_result(
        result,
        experiment=experiment,
        expected_config=expected_config,
        expected_seed=seed,
        expected_run_context=expected_run_context,
    )


def env_tag(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"
