"""Hindsight goal-conditioning helpers for sparse classic-control tasks."""

from __future__ import annotations

import numpy as np


def append_goal(obs: np.ndarray, goal: np.ndarray | float) -> np.ndarray:
    """Concatenate an observation with a scalar or vector goal."""
    obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    goal_arr = np.asarray(goal, dtype=np.float32).reshape(-1)
    return np.concatenate([obs_arr, goal_arr]).astype(np.float32)


def mountaincar_achieved_goal(obs: np.ndarray) -> np.ndarray:
    """Use car position as the achieved goal coordinate."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    return np.array([obs_arr[0]], dtype=np.float32)


def acrobot_end_effector_height(obs: np.ndarray) -> np.ndarray:
    """Return the Acrobot end-effector height used by the terminal condition."""
    obs_arr = np.asarray(obs, dtype=np.float32)
    cos_1, sin_1, cos_2, sin_2 = obs_arr[:4]
    cos_12 = cos_1 * cos_2 - sin_1 * sin_2
    return np.array([-(cos_1 + cos_12)], dtype=np.float32)


def goal_reached(achieved: np.ndarray, goal: np.ndarray, tolerance: float = 1e-3) -> bool:
    """Return whether an achieved goal is within tolerance of a desired goal."""
    achieved_arr = np.asarray(achieved, dtype=np.float32).reshape(-1)
    goal_arr = np.asarray(goal, dtype=np.float32).reshape(-1)
    if achieved_arr.shape != goal_arr.shape:
        raise ValueError(
            f"achieved and goal shapes must match, got {achieved_arr.shape} and {goal_arr.shape}"
        )
    return bool(np.linalg.norm(achieved_arr - goal_arr) <= tolerance)


def hindsight_event_targets(
    achieved_goals: list[np.ndarray],
    desired_goal: np.ndarray,
    *,
    gamma: float,
    horizon: int,
    tolerance: float = 1e-3,
    success_value: float = 1.0,
) -> list[float]:
    """Create finite-horizon event targets for a desired hindsight goal."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")
    targets = np.zeros(len(achieved_goals), dtype=np.float64)
    for event_idx, achieved in enumerate(achieved_goals):
        if not goal_reached(achieved, desired_goal, tolerance=tolerance):
            continue
        start = max(0, event_idx - horizon + 1)
        for step in range(start, event_idx + 1):
            targets[step] += success_value * (gamma ** (event_idx - step))
    return [float(v) for v in targets]


def terminal_hindsight_goal(achieved_goals: list[np.ndarray]) -> np.ndarray:
    """Use the final achieved goal as a hindsight desired goal."""
    if not achieved_goals:
        raise ValueError("achieved_goals must not be empty")
    return np.asarray(achieved_goals[-1], dtype=np.float32).reshape(-1)
