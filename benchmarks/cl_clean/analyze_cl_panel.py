"""Panel analyzer for the cl_clean continual-learning factorial sweep.

Cells (up to 7): ``nef_accumulate``, ``nef_joint``, ``nef_reset``,
``mlp_finetune``, ``mlp_joint``, ``mlp_ewc``, ``mlp_er``.  Reports:

* per-cell ``mean ± SE`` of ``test_accuracy`` and ``fit_time_s`` for
  every cell present in the panel JSON;
* the **joint-equivalence gap** — the per-seed signed difference
  ``test_accuracy[nef_accumulate] - test_accuracy[nef_joint]`` and its
  max-abs over seeds (Paper II H1 sentinel: ``≤ 0.05`` pp);
* the **forgetting gap** — the per-seed difference
  ``test_accuracy[nef_accumulate] - test_accuracy[nef_reset]`` (Paper II
  H1 punchline: large positive gap ⇒ accumulate retains old tasks the
  reset baseline drops);
* per-task accuracy curves where present.

Reproducibility: each row is checked against the supplied
``--base-config`` for ``code_identity``, ``runner_version``,
``run_context``, ``seed_policy`` and configuration consistency.  Rows
that do not match are listed in the verdict's ``stale_rows`` field and
excluded from aggregates.  This refuses to silently mix runs from
different code states (the Paper-I ``code_identity`` discipline).

Never mutates the base config.  Optional ``--out`` writes the verdict
as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

from .cl_repro import (
    RESULT_NAMESPACE,
    RUNNER_VERSION,
    code_identity,
    seed_policy,
)
from .experiments import ALL_CL_CELLS, NEF_CL_CELLS
from .experiments import get as get_experiment


def _row_validity(row: dict, *, experiment: str, base_config: dict | None) -> tuple[bool, str]:
    """Return (is_valid_data_row, reason_if_not)."""
    if not isinstance(row, dict):
        return False, "not a dict"
    if "error" in row:
        return False, f"error row: {row.get('error')}"
    if row.get("result_namespace") != RESULT_NAMESPACE:
        return False, f"namespace mismatch ({row.get('result_namespace')!r})"
    if row.get("runner_version") != RUNNER_VERSION:
        return False, f"runner_version mismatch ({row.get('runner_version')!r})"
    if row.get("experiment") != experiment:
        return False, f"experiment mismatch ({row.get('experiment')!r})"
    if "metrics" not in row:
        return False, "no metrics"
    expected_id = code_identity()
    if expected_id is not None and row.get("code_identity") != expected_id:
        return False, f"code_identity mismatch (row={row.get('code_identity')!r})"
    if row.get("seed_policy") != seed_policy():
        return False, "seed_policy mismatch"
    if base_config is not None:
        knob = row.get("knob")
        value = row.get("value")
        cfg = row.get("config", {})
        if knob is None or knob not in base_config:
            return False, f"knob {knob!r} not in base_config"
        # Every key in base_config (other than the swept knob) must match.
        for k, v in base_config.items():
            if k == knob:
                continue
            if cfg.get(k) != v:
                return False, f"config mismatch on {k!r}: row={cfg.get(k)!r} base={v!r}"
        if cfg.get(knob) != value:
            return False, f"value mismatch: knob={knob} cfg={cfg.get(knob)!r} value={value!r}"
        spec = get_experiment(experiment)
        expected_ctx = dict(spec.default_run_context())
        row_ctx = row.get("run_context", {})
        for k, v in expected_ctx.items():
            if row_ctx.get(k) != v:
                return False, f"run_context mismatch on {k!r}"
    return True, ""


def _se(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def aggregate(rows: list[dict]) -> dict:
    by_cell: dict[str, dict[int, dict]] = {c: {} for c in ALL_CL_CELLS}
    for r in rows:
        cfg = r.get("config", {})
        cell = cfg.get("cell")
        seed = r.get("seed")
        if cell not in by_cell or seed is None:
            continue
        m = r.get("metrics", {})
        by_cell[cell][int(seed)] = {
            "acc": float(m.get("test_accuracy", float("nan"))),
            "fit_time_s": float(m.get("fit_time_s", float("nan"))),
            "per_task_accuracy": m.get("per_task_accuracy", []),
        }

    out: dict = {"by_cell": {}}
    for cell, by_seed in by_cell.items():
        accs = [v["acc"] for v in by_seed.values()]
        fts = [v["fit_time_s"] for v in by_seed.values()]
        out["by_cell"][cell] = {
            "n": len(accs),
            "mean_acc": mean(accs) if accs else float("nan"),
            "se_acc": _se(accs),
            "mean_fit_time_s": mean(fts) if fts else float("nan"),
            "se_fit_time_s": _se(fts),
            "by_seed": by_seed,
        }

    # Joint-equivalence and forgetting gap are scoped to the NEF cells.
    common_seeds = (
        set(by_cell["nef_accumulate"]).intersection(by_cell["nef_joint"])
        if by_cell["nef_accumulate"] and by_cell["nef_joint"]
        else set()
    )
    eq_gaps = [
        by_cell["nef_accumulate"][s]["acc"] - by_cell["nef_joint"][s]["acc"]
        for s in sorted(common_seeds)
    ]
    out["joint_equivalence"] = {
        "n_seeds": len(eq_gaps),
        "mean_signed_gap": mean(eq_gaps) if eq_gaps else float("nan"),
        "max_abs_gap": max((abs(g) for g in eq_gaps), default=float("nan")),
        "se_signed_gap": _se(eq_gaps),
        "verdict": _equivalence_verdict(eq_gaps),
    }

    common_seeds_r = set(by_cell["nef_accumulate"]).intersection(by_cell["nef_reset"])
    forget = [
        by_cell["nef_accumulate"][s]["acc"] - by_cell["nef_reset"][s]["acc"]
        for s in sorted(common_seeds_r)
    ]
    out["forgetting"] = {
        "n_seeds": len(forget),
        "mean_gap": mean(forget) if forget else float("nan"),
        "se_gap": _se(forget),
    }
    out["nef_cells"] = list(NEF_CL_CELLS)
    out["all_cells"] = list(ALL_CL_CELLS)
    return out


def _equivalence_verdict(gaps: list[float]) -> str:
    if not gaps:
        return "no data"
    max_abs = max(abs(g) for g in gaps)
    if max_abs <= 0.0005:  # 0.05 pp on [0, 1] accuracy scale
        return "joint-equivalence supported (≤ 0.05 pp)"
    if max_abs <= 0.005:
        return "joint-equivalence approximate (≤ 0.5 pp)"
    return "joint-equivalence violated (> 0.5 pp)"


def _format(out: dict) -> str:
    lines = []
    lines.append("Per-cell summary:")
    for cell, s in out["by_cell"].items():
        if s["n"] == 0:
            lines.append(f"  {cell:<18} n=0")
            continue
        lines.append(
            f"  {cell:<18} n={s['n']:>2}  acc={100 * s['mean_acc']:.3f}% ± {100 * s['se_acc']:.3f}%"
            f"  fit={s['mean_fit_time_s']:.2f}s ± {s['se_fit_time_s']:.2f}s"
        )
    je = out["joint_equivalence"]
    lines.append("")
    lines.append(
        "Joint-equivalence (accumulate − joint): "
        f"mean={100 * je['mean_signed_gap']:+.4f} pp  "
        f"max|Δ|={100 * je['max_abs_gap']:.4f} pp  "
        f"n={je['n_seeds']}  →  {je['verdict']}"
    )
    fg = out["forgetting"]
    lines.append(
        "Forgetting gap   (accumulate − reset):  "
        f"mean={100 * fg['mean_gap']:+.3f} pp ± {100 * fg['se_gap']:.3f} pp  n={fg['n_seeds']}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--sweep-json", required=True, type=Path)
    p.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Optional base config JSON; rows whose config or run_context "
        "diverge are listed as stale and excluded from aggregates.",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any stale rows are found.",
    )
    args = p.parse_args(argv)

    sweep = json.loads(args.sweep_json.read_text())
    base_config: dict | None = None
    if args.base_config is not None:
        base_config = json.loads(args.base_config.read_text())

    rows: list[dict] = []
    stale: list[dict] = []
    for key, row in sweep.items():
        ok, reason = _row_validity(row, experiment=args.experiment, base_config=base_config)
        if ok:
            rows.append(row)
        elif isinstance(row, dict) and "metrics" in row and "error" not in row:
            stale.append({"key": key, "reason": reason})
    if not rows:
        print(f"no valid data rows for experiment {args.experiment!r}", file=sys.stderr)
        if stale:
            print(f"  ({len(stale)} stale rows ignored)", file=sys.stderr)
        return 1
    out = aggregate(rows)
    out["stale_rows"] = stale
    out["expected_code_identity"] = code_identity()
    print(_format(out))
    if stale:
        print(f"\nWARNING: {len(stale)} stale rows excluded from aggregates:")
        for s in stale[:10]:
            print(f"  {s['key']}: {s['reason']}")
        if len(stale) > 10:
            print(f"  ... and {len(stale) - 10} more")
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    if args.strict and stale:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
