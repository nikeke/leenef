"""Panel analyzer for the cl_clean NEF-CL factorial sweep.

Cells: ``nef_accumulate``, ``nef_joint``, ``nef_reset``.  Reports:

* per-cell ``mean ± SE`` of ``test_accuracy`` and ``fit_time_s``;
* the **joint-equivalence gap** — the per-seed signed difference
  ``test_accuracy[nef_accumulate] - test_accuracy[nef_joint]`` and its
  max-abs over seeds (Paper II H1 sentinel: ``≤ 0.05`` pp);
* the **forgetting gap** — the per-seed difference
  ``test_accuracy[nef_accumulate] - test_accuracy[nef_reset]`` (Paper II
  H1 punchline: large positive gap ⇒ accumulate retains old tasks the
  reset baseline drops);
* per-task accuracy curves where present.

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

from .cl_repro import RESULT_NAMESPACE
from .experiments import NEF_CL_CELLS


def _is_data_row(row: dict, *, experiment: str) -> bool:
    return (
        isinstance(row, dict)
        and "error" not in row
        and row.get("result_namespace") == RESULT_NAMESPACE
        and row.get("experiment") == experiment
        and "metrics" in row
    )


def _se(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def aggregate(rows: list[dict]) -> dict:
    by_cell: dict[str, dict[int, dict]] = {c: {} for c in NEF_CL_CELLS}
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
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    sweep = json.loads(args.sweep_json.read_text())
    rows = [r for r in sweep.values() if _is_data_row(r, experiment=args.experiment)]
    if not rows:
        print(f"no data rows for experiment {args.experiment!r}", file=sys.stderr)
        return 1
    out = aggregate(rows)
    print(_format(out))
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
