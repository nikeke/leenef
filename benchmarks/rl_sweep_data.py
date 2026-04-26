"""Shared RL sweep data-loading helpers used by analysis, plotting, and tests."""

from __future__ import annotations

import json
import os

from benchmarks.rl_repro import is_current_hp_sweep_result, result_robust_eval


def load_sweep(
    path: str,
    knob: str,
    *,
    base_config: dict[str, object],
    expected_run_context: dict[str, object],
) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    by_value: dict = {}
    for result in data.values():
        if not is_current_hp_sweep_result(
            result,
            base_config=base_config,
            knob=knob,
            expected_run_context=expected_run_context,
        ):
            continue
        robust_eval = result_robust_eval(result)
        if robust_eval is None:
            continue
        best_eval = result.get("best_eval")
        value = result.get("value")
        key = (
            (float("inf"), "none")
            if value is None
            else (value if isinstance(value, (int, float)) else 0, str(value))
        )
        by_value.setdefault(key, {"value": value, "robust": [], "best": [], "final": []})
        by_value[key]["robust"].append(robust_eval)
        if best_eval is not None:
            by_value[key]["best"].append(best_eval)
        final_eval = result.get("final_eval")
        if final_eval is not None:
            by_value[key]["final"].append(final_eval)
    return dict(sorted(by_value.items()))
