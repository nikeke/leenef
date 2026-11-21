"""Merge current rows across panel files and summarize prespecified paired contrasts."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import fmean, stdev

from benchmarks.sl_clean.sl_repro import code_identity

from .common import atomic_write, current_rows, read_results

try:
    from scipy.stats import t as student_t
except ImportError:
    student_t = None

STRATEGY = {"global": "hypersphere", "patch": "receptive_field"}


def _summary(values: list[float]) -> dict:
    n = len(values)
    return {
        "n": n,
        "mean": fmean(values) if n else None,
        "se": stdev(values) / math.sqrt(n) if n > 1 else None,
    }


def _t_survival(t: float, df: int) -> float:
    if student_t is not None:
        return float(student_t.sf(t, df))
    if t < 0:
        return 1 - _t_survival(-t, df)
    # t = sqrt(df) * tan(theta) makes the infinite interval finite.
    start = math.atan(t / math.sqrt(df))
    scale = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(math.pi)
    n = 4096
    step = (math.pi / 2 - start) / n
    area = math.fsum(
        (4 if i % 2 else 2) * math.cos(start + i * step) ** (df - 1) for i in range(1, n)
    )
    return (
        scale * step / 3 * (math.cos(start) ** (df - 1) + area + math.cos(math.pi / 2) ** (df - 1))
    )


@lru_cache(maxsize=None)
def _t_critical(df: int) -> float:
    if student_t is not None:
        return float(student_t.ppf(0.95, df))
    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _t_survival(mid, df) > 0.05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _paired(lhs: dict[int, float], rhs: dict[int, float], *, inference: bool = False) -> dict:
    out = _summary([lhs[seed] - rhs[seed] for seed in sorted(lhs.keys() & rhs.keys())])
    mean, se, n = out["mean"], out["se"], out["n"]
    ratio = None
    if se:
        ratio = mean / se
    elif se == 0:
        ratio = "inf" if mean > 0 else "-inf" if mean < 0 else 0.0
    if not inference:
        return {**out, "z": ratio}
    return {
        **out,
        "t": ratio,
        "p_one_sided": _t_survival(float(ratio), n - 1) if ratio is not None else None,
        "lower_95": mean - _t_critical(n - 1) * se if n > 1 else None,
    }


def _accuracy(
    rows: list[dict], stage: str | None = None, metric: str = "test_accuracy"
) -> dict[int, float]:
    out = {}
    for row in rows:
        source = row if stage is None else row.get(stage, {})
        if metric in source and "seed" in row:
            out[row["seed"]] = source[metric]
    return out


def _strategy(row: dict) -> str:
    return row.get("encoder_strategy", row.get("strategy", ""))


def _protocol(row: dict) -> tuple:
    return row.get("val_size"), row.get("fit_limit")


def _reference(prune: list[dict], metric: str) -> dict:
    grouped = defaultdict(list)
    for row in prune:
        if row.get("criterion") == "random" and row.get("P") is None:
            grouped[(row["dataset"], _protocol(row), _strategy(row), row["k"])].append(row)
    return {key: _accuracy(rows, metric=metric) for key, rows in grouped.items()}


def _equivalent_width(
    curve: dict[int, dict[int, float]], target: float | None
) -> str | float | None:
    points = [(width, fmean(scores.values())) for width, scores in sorted(curve.items()) if scores]
    if not points or target is None:
        return None
    if target <= points[0][1]:
        return f"<{points[0][0]}" if target < points[0][1] else points[0][0]
    for (w0, a0), (w1, a1) in zip(points, points[1:]):
        if a0 < target <= a1:
            return round(w0 * (w1 / w0) ** ((target - a0) / (a1 - a0)), 1)
    return f">{points[-1][0]}"


def summarize(
    rows: list[dict], prune_rows: list[dict], exploratory: set[str], metric: str = "test_accuracy"
) -> dict:
    grouped = defaultdict(list)
    refs = _reference(prune_rows, metric)
    for row in rows:
        if "criterion" in row:
            key = ("prune", row["dataset"], _protocol(row), _strategy(row), row.get("P"), row["k"])
        elif "family" in row:
            key = (
                "growth",
                row["dataset"],
                _protocol(row),
                row["family"],
                row.get("n_generations"),
                row["k"],
            )
        else:
            continue
        grouped[key].append(row)

    report = {"prune": [], "growth": [], "contrasts": [], "diagnostics": []}
    for (panel, dataset, protocol, group, level, k), group_rows in grouped.items():
        base = {
            "dataset": dataset,
            "k": k,
            "val_size": protocol[0],
            "fit_limit": protocol[1],
            "exploratory": dataset in exploratory,
        }
        if panel == "prune":
            baseline = refs.get((dataset, protocol, group, k), {})
            for criterion in sorted({row["criterion"] for row in group_rows}):
                acc = _accuracy(
                    [row for row in group_rows if row["criterion"] == criterion], metric=metric
                )
                summary = {
                    **base,
                    "encoder_strategy": group,
                    "P": level,
                    "criterion": criterion,
                    "test": _summary(list(acc.values())),
                    "vs_random": _paired(acc, baseline),
                }
                report["prune"].append(summary)
                if level == 4000 and k == 1000 and criterion == "obs":
                    report["contrasts"].append(
                        {
                            **base,
                            "panel": "prune",
                            "encoder_strategy": group,
                            "P": level,
                            "contrast": "obs − random",
                            **_paired(acc, baseline, inference=True),
                        }
                    )
            continue

        base.update(family=group, n_generations=level)
        arm_rows = {
            arm: [row for row in group_rows if row["arm"] == arm]
            for arm in sorted({row["arm"] for row in group_rows})
        }
        finals = {arm: _accuracy(cells, "final", metric) for arm, cells in arm_rows.items()}
        baseline = finals.get("random", {})
        curve = {
            width: acc
            for (ds, p, strategy, width), acc in refs.items()
            if ds == dataset and p == protocol and strategy == STRATEGY.get(group)
        }
        for arm, cells in arm_rows.items():
            per_gen = defaultdict(dict)
            for row in cells:
                for gen in row.get("generations", []):
                    if metric in gen:
                        per_gen[gen["gen"]][row["seed"]] = gen[metric]
            selected = _summary(list(_accuracy(cells, "val_selected", metric).values()))
            report["growth"].append(
                {
                    **base,
                    "arm": arm,
                    "gen0": _summary(list(per_gen[0].values())),
                    "final": _summary(list(finals[arm].values())),
                    "val_selected": selected,
                    "vs_random": _paired(finals[arm], baseline),
                    "per_generation": {
                        g: _summary(list(scores.values())) for g, scores in sorted(per_gen.items())
                    },
                    "equivalent_random_width": _equivalent_width(curve, selected["mean"]),
                }
            )

        directed = "phd" if group == "global" else "phd_patch"
        if directed not in finals:
            continue
        comparisons = {"width3x": curve.get(3 * k, {}), "proposal": baseline}
        if group == "global":
            comparisons.update(
                mechanism=finals.get("phd_classshuffle", {}),
                selection=finals.get("phd_randkeep", {}),
            )
        for contrast, control in comparisons.items():
            report["contrasts"].append(
                {
                    **base,
                    "panel": "growth",
                    "contrast": contrast,
                    **_paired(finals[directed], control, inference=True),
                }
            )
        if group == "global" and "phd_classshuffle" in arm_rows:
            mse = [
                {
                    row["seed"]: gen["val_mse_loop"]
                    for row in arm_rows[arm]
                    for gen in row.get("generations", [])
                    if gen["gen"] == 1 and "val_mse_loop" in gen
                }
                for arm in ("phd_classshuffle", "phd")
            ]
            report["diagnostics"].append(
                {
                    **base,
                    "diagnostic": "gen-1 val_mse_loop: phd_classshuffle − phd",
                    **_summary(
                        [mse[0][s] - mse[1][s] for s in sorted(mse[0].keys() & mse[1].keys())]
                    ),
                }
            )
    return report


def _holm(contrasts: list[dict]) -> None:
    family = [
        row
        for row in contrasts
        if row["k"] == 1000
        and row.get("family") == "global"
        and not row["exploratory"]
        and row["contrast"] in ("width3x", "proposal", "mechanism")
        and row["p_one_sided"] is not None
        and row.get("fit_limit") is None
    ]
    unique = {}
    for row in family:
        key = row["dataset"], row["contrast"]
        if key not in unique or (row.get("n_generations") or 0) > (
            unique[key].get("n_generations") or 0
        ):
            unique[key] = row
    adjusted = 0.0
    for i, row in enumerate(sorted(unique.values(), key=lambda row: row["p_one_sided"])):
        adjusted = max(adjusted, min(1.0, row["p_one_sided"] * (len(unique) - i)))
        for duplicate in family:
            if duplicate == row or (
                duplicate["dataset"] == row["dataset"]
                and duplicate["contrast"] == row["contrast"]
                and duplicate.get("n_generations") == row.get("n_generations")
                and duplicate["p_one_sided"] == row["p_one_sided"]
            ):
                duplicate["p_holm"] = adjusted
                duplicate["decision"] = "pass" if adjusted <= 0.05 else "fail"


def _format(value: float | str | None) -> str:
    return "—" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)


def _mean_se(stats: dict) -> str:
    return f"{_format(stats['mean'])} ± {_format(stats['se'])} (n={stats['n']})"


def _table(heading: str, columns: tuple[str, ...], rows: list[tuple]) -> None:
    if not rows:
        return
    print(f"\n### {heading}\n")
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(_format(value) for value in row) + " |")


def print_summary(summary: dict) -> None:
    def dataset(row: dict) -> str:
        return row["dataset"] + (" (exploratory)" if row["exploratory"] else "")

    _table(
        "Pruning",
        ("Dataset", "Strategy", "P", "k", "Criterion", "Test", "Δ random", "z"),
        [
            (
                dataset(r),
                r["encoder_strategy"],
                r["P"],
                r["k"],
                r["criterion"],
                _mean_se(r["test"]),
                _mean_se(r["vs_random"]),
                r["vs_random"]["z"],
            )
            for r in summary["prune"]
        ],
    )
    _table(
        "Growth",
        (
            "Dataset",
            "Family",
            "k",
            "Arm",
            "Gen 0",
            "Final",
            "Val-selected",
            "Final Δ random",
            "Equivalent width",
            "Mean test by generation",
        ),
        [
            (
                dataset(r),
                r["family"],
                r["k"],
                r["arm"],
                _mean_se(r["gen0"]),
                _mean_se(r["final"]),
                _mean_se(r["val_selected"]),
                _mean_se(r["vs_random"]),
                r["equivalent_random_width"],
                ", ".join(f"{g}:{_format(s['mean'])}" for g, s in r["per_generation"].items()),
            )
            for r in summary["growth"]
        ],
    )
    for primary, section, heading in (
        (
            True,
            "contrasts",
            "Primary paired contrasts: `test_accuracy` (validation-selected alpha)",
        ),
        (
            False,
            "contrasts_loop",
            "Secondary paired contrasts: `test_accuracy_loop` (fixed loop alpha)",
        ),
    ):
        _table(
            f"{heading}; final generation for growth",
            (
                "Dataset",
                "Family/strategy",
                "P",
                "k",
                "Contrast",
                "n",
                "Mean Δ",
                "SE",
                "t",
                "One-sided p",
                "Unadjusted lower 95%",
                *(("Holm p", "Decision") if primary else ()),
            ),
            [
                (
                    dataset(r),
                    r.get("family", r.get("encoder_strategy")),
                    r.get("P"),
                    r["k"],
                    r["contrast"],
                    r["n"],
                    r["mean"],
                    r["se"],
                    r["t"],
                    r["p_one_sided"],
                    r["lower_95"],
                    *((r.get("p_holm"), r.get("decision")) if primary else ()),
                )
                for r in summary[section]
            ],
        )
    failures = summary["eig_failures"]
    print("\n### Subspace convergence\n")
    if failures:
        print(f"{len(failures)} growth row(s) have non-converged generations:")
        for row in failures:
            generations = ", ".join(map(str, row["generations"]))
            print(
                f"- {row['dataset']} / {row['family']} / {row['arm']}, "
                f"k={row['k']}, seed={row['seed']}: {generations}"
            )
    else:
        print("All converged (among recorded subspace solves).")
    _table(
        "Diagnostic (no test)",
        ("Dataset", "k", "Contrast", "Δ"),
        [(dataset(r), r["k"], r["diagnostic"], _mean_se(r)) for r in summary["diagnostics"]],
    )


def _merge_rows(files: list[Path], loaded: dict[Path, list[dict]]) -> list[dict]:
    rows, seen = [], set()
    for path in dict.fromkeys(files):
        for row in loaded[path]:
            key = row.get("key")
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            rows.append(row)
    return rows


def _eig_failures(rows: list[dict]) -> list[dict]:
    failures = []
    for row in rows:
        generations = [
            gen["gen"] for gen in row.get("generations", []) if gen.get("eig_converged") is False
        ]
        if generations:
            fields = ("dataset", "family", "arm", "k", "seed")
            failures.append(
                {**{field: row[field] for field in fields}, "generations": generations}
            )
    return failures


def _file_summary(summary: dict, source: list[dict]) -> dict:
    panels = {
        panel: {row["dataset"] for row in source if "dataset" in row and discriminator in row}
        for panel, discriminator in (("growth", "family"), ("prune", "criterion"))
    }

    def belongs(section: str, row: dict) -> bool:
        panel = row.get("panel", "prune" if section == "prune" else "growth")
        return row["dataset"] in panels[panel]

    return {
        section: [row for row in rows if belongs(section, row)]
        for section, rows in summary.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="prune panels or growth shards")
    parser.add_argument(
        "--prune-json", nargs="+", type=Path, default=[], help="additional pruning panels"
    )
    parser.add_argument("--identity")
    parser.add_argument("--exploratory", nargs="+", default=[], metavar="DATASET")
    args = parser.parse_args(argv)
    identity = args.identity or code_identity()
    if identity is None:
        parser.error("no Git identity available; pass --identity")
    paths = [*args.files, *args.prune_json]
    loaded = {path: current_rows(read_results(path), identity) for path in dict.fromkeys(paths)}
    rows = _merge_rows(paths, loaded)
    prune = [row for row in rows if "criterion" in row]
    exploratory = set(args.exploratory)
    summary = summarize(rows, prune, exploratory)
    summary["contrasts_loop"] = summarize(rows, prune, exploratory, "test_accuracy_loop")[
        "contrasts"
    ]
    summary["eig_failures"] = _eig_failures(rows)
    _holm(summary["contrasts"])
    print(f"\n## Neuron selection ({identity}; {len(rows)} unique current rows)\n")
    print_summary(summary)
    for path in dict.fromkeys(args.files):
        atomic_write(
            path.with_name(f"{path.stem}_summary.json"),
            {"code_identity": identity, **_file_summary(summary, loaded[path])},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
