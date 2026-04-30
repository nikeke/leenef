"""Shared reproducibility helpers for supervised-learning clean sweeps.

This is the SL counterpart to ``benchmarks/rl_repro.py`` (archived under
``attic/benchmarks/rl_repro.py``).  The key contract:

* every result row stores enough metadata that the runner can decide
  whether it was produced by *the current code, current configuration,
  current run context, and current seed* — and skip it on resume if so;
* ``code_identity`` collapses to the HEAD SHA for clean trees and to
  ``<sha>-dirty-<hash>`` for dirty trees, while ignoring generated
  artifacts (figures, results) so the hash is stable across reruns.

Per-experiment paths and run-context schemas live in the experiment
registry (``benchmarks.sl_clean.experiments``) — this module is
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

RESULT_NAMESPACE = "sl_clean"
RUNNER_VERSION = "sl-clean-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / RESULT_NAMESPACE
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
DEFAULT_BASE_CONFIG_PATH = RESULTS_DIR / "base_config.json"
DOC_PATH = REPO_ROOT / "docs" / "sl_clean_sweeps.md"

# Files/paths that participate in the code-identity hash.  Generated
# artifacts (results/, docs/figures/, attic/) are deliberately excluded
# so reruns that only update those do not invalidate cached cells.
CODE_IDENTITY_PATHS: tuple[str, ...] = (
    "src",
    "benchmarks",
    "tests",
    "pyproject.toml",
)


def set_sl_seed(seed: int | None) -> None:
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
    """User-facing description of how the per-run seed is consumed."""
    return {
        "main_seed": "per-run user-visible seed",
        "torch_seed": "torch.manual_seed(seed)",
        "numpy_seed": "np.random.seed(seed)",
        "python_seed": "random.seed(seed)",
        "data_split_seed": "derived deterministically from seed inside the experiment",
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
    """Identity for the source tree, ignoring generated artifacts.

    Clean tree → ``<HEAD-sha>``.
    Dirty tree → ``<HEAD-sha>-dirty-<sha256[:12]>``, where the inner hash
    is computed from the diff of ``CODE_IDENTITY_PATHS`` plus the list of
    untracked files in the same paths.
    """
    head_out = _git("rev-parse", "HEAD")
    if not head_out:
        return None
    head = head_out.strip()
    if not head:
        return None
    diff = _git("diff", "--no-ext-diff", "HEAD", "--", *CODE_IDENTITY_PATHS) or ""
    untracked = (
        _git("ls-files", "--others", "--exclude-standard", "--", *CODE_IDENTITY_PATHS) or ""
    )
    if not diff and not untracked:
        return head
    blob = f"{diff}\n--untracked--\n{untracked}".encode("utf-8")
    dirty = hashlib.sha256(blob).hexdigest()[:12]
    return f"{head}-dirty-{dirty}"


def reset_code_identity_cache() -> None:
    """Clear the cached identity (use in tests that mutate the tree)."""
    code_identity.cache_clear()


def hardware_info() -> dict[str, object]:
    """Lightweight hardware fingerprint included in every result row."""
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
    """Skeleton metadata stamped onto every result row.

    Experiment-specific metric fields are added by the runner *on top* of
    this skeleton.  The fields here are exactly what ``can_reuse_result``
    consults when deciding whether to skip a cell on resume.
    """
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
    """Decide whether a stored row was produced by the current code/config.

    A row is reusable iff it carries no error, lives in our namespace at
    the current runner version, was produced by the current code identity,
    and matches the requested experiment / config / seed / run context.
    """
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
    """Return ``base_config`` with a single knob overridden — sweep contract."""
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
    """Predicate used by analyzers to filter a stored sweep file.

    Only rows that match the current base config (with this knob varied)
    and the current run context, code identity, and runner version are
    considered when computing aggregates and adoption decisions.
    """
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
    """Filesystem-safe lowercase slug for dataset / experiment names."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"
