"""Shared reproducibility helpers for RL benchmarks."""

from __future__ import annotations

import hashlib
import random
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

RESULT_NAMESPACE = "rl_clean"
RESULTS_DIR = Path("results") / RESULT_NAMESPACE
FIGURES_DIR = RESULTS_DIR / "figures"
BASE_CONFIG_PATH = RESULTS_DIR / "base_config.json"
DOC_PATH = Path("docs") / "rl_clean_sweeps.md"

RUNNER_VERSION = "rl-clean-v1"
WARMUP_SEED_OFFSET = 100_000
EVAL_SEED_BASE = 10_000


def set_rl_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic worker execution."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def warmup_seed(seed: int) -> int:
    return WARMUP_SEED_OFFSET + seed


def training_seed(seed: int, episode: int) -> int:
    return seed + episode


def evaluation_seed(base: int, episode: int = 0) -> int:
    return base + episode


def reset_env(env, *, seed: int | None):
    """Reset an environment and seed its action space when applicable."""
    if seed is not None:
        action_space = getattr(env, "action_space", None)
        if action_space is not None and hasattr(action_space, "seed"):
            action_space.seed(seed)
        return env.reset(seed=seed)
    return env.reset()


def seed_policy() -> dict[str, object]:
    return {
        "main_seed": "per-run user-visible seed",
        "warmup_env_seed": f"{WARMUP_SEED_OFFSET} + seed + episode",
        "training_env_seed": "seed + episode",
        "evaluation_env_seed": f"{EVAL_SEED_BASE} + episode",
        "action_space_seed": "matches env.reset seed when provided",
    }


@lru_cache(maxsize=1)
def code_version() -> str | None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    if not head:
        return None
    try:
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return head
    if not diff and not untracked:
        return head
    dirty_hash = hashlib.sha256(f"{diff}\n--untracked--\n{untracked}".encode("utf-8")).hexdigest()[
        :12
    ]
    return f"{head}-dirty-{dirty_hash}"


def result_metadata(
    config: dict,
    seed: int,
    *,
    run_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Metadata required to compare or resume RL runs safely."""
    return {
        "result_namespace": RESULT_NAMESPACE,
        "runner_version": RUNNER_VERSION,
        "config": config,
        "seed": seed,
        "seed_policy": seed_policy(),
        "code_version": code_version(),
        "run_context": run_context or {},
    }


def hp_run_context(
    env_name: str,
    n_episodes: int,
    *,
    eval_every: int = 50,
    eval_episodes: int = 20,
) -> dict[str, object]:
    return {
        "env_name": env_name,
        "n_episodes": n_episodes,
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
    }


def hp_expected_config(base_config: dict[str, object], knob: str, value) -> dict[str, object]:
    expected = dict(base_config)
    expected[knob] = value
    return expected


def env_tag(env_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", env_name.lower()).strip("_")


def can_reuse_result(
    result: dict,
    expected_config: dict,
    *,
    expected_seed: int,
    expected_run_context: dict[str, object] | None = None,
) -> bool:
    """Return True when a stored result matches the current clean runner semantics."""
    return (
        "error" not in result
        and result.get("result_namespace") == RESULT_NAMESPACE
        and result.get("runner_version") == RUNNER_VERSION
        and result.get("config") == expected_config
        and result.get("seed") == expected_seed
        and result.get("seed_policy") == seed_policy()
        and result.get("code_version") == code_version()
        and result.get("run_context", {}) == (expected_run_context or {})
    )


def is_current_hp_sweep_result(
    result: dict,
    *,
    base_config: dict[str, object],
    knob: str,
    expected_run_context: dict[str, object],
) -> bool:
    if "error" in result or result.get("knob") != knob:
        return False
    if "seed" not in result:
        return False
    return can_reuse_result(
        result,
        hp_expected_config(base_config, knob, result.get("value")),
        expected_seed=result["seed"],
        expected_run_context=expected_run_context,
    )
