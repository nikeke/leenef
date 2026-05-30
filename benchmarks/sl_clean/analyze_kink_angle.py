"""Aggregate the kink-angle sweep into mean +/- SE per angle per dataset.

Reads results/sl_clean/kink_angle_{dataset}_panel.json (filtering to rows
whose code_identity matches HEAD) and writes a compact summary JSON plus a
printed table.  This is the analyzer behind docs/activation_kernel_theory.md
section 8.2.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .sl_repro import RESULTS_DIR, code_identity


def _aggregate(path: Path, require_identity: bool) -> dict:
    with open(path) as f:
        rows = json.load(f)
    head = code_identity()
    by_angle: dict[float, list[float]] = defaultdict(list)
    identities = set()
    for r in rows.values():
        if not isinstance(r, dict) or "theta_over_pi" not in r:
            continue
        identities.add(r.get("code_identity"))
        if require_identity and r.get("code_identity") != head:
            continue
        by_angle[r["theta_over_pi"]].append(r["test_accuracy"])
    summary = {}
    for tp in sorted(by_angle):
        accs = by_angle[tp]
        m = sum(accs) / len(accs)
        var = sum((a - m) ** 2 for a in accs) / max(1, len(accs) - 1)
        se = math.sqrt(var) / math.sqrt(len(accs))
        summary[f"{tp:.3f}"] = {
            "theta_over_pi": tp,
            "n_seeds": len(accs),
            "mean": m,
            "se": se,
        }
    return {"summary": summary, "identities_present": sorted(i for i in identities if i)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--datasets", default="mnist,fashion,cifar10")
    ap.add_argument("--all-identities", action="store_true")
    args = ap.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out = {}
    for d in datasets:
        path = RESULTS_DIR / f"kink_angle_{d}_panel.json"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        agg = _aggregate(path, require_identity=not args.all_identities)
        out[d] = agg["summary"]
        print(
            f"\n=== {d}  (code_identity match={not args.all_identities}; "
            f"identities={agg['identities_present']}) ==="
        )
        print(f"{'theta/pi':>9}  {'n':>3}  {'mean':>8}  {'se':>7}")
        for tp, s in agg["summary"].items():
            tag = ""
            if abs(s["theta_over_pi"] - 0.50) < 1e-9:
                tag = " abs"
            elif abs(s["theta_over_pi"] - 0.75) < 1e-9:
                tag = " relu"
            elif abs(s["theta_over_pi"] - 1.00) < 1e-9:
                tag = " LINEAR(collapse)"
            print(f"{tp:>9}  {s['n_seeds']:>3}  {s['mean']:>8.4f}  {s['se']:>7.4f}{tag}")

    out_path = RESULTS_DIR / "kink_angle_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\n[kink_angle] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
