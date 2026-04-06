"""Summarize and compare Colab benchmark result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    """Load either BenchmarkResult JSON or smoke-test JSON."""
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    raise ValueError(f"Unsupported result format: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare Colab benchmark result files")
    parser.add_argument("paths", nargs="+", type=Path, help="Result JSON files to summarize")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline JSON to compare times and accuracy against",
    )
    return parser.parse_args(argv)


def make_baseline_map(path: Path | None) -> dict[tuple[str, str], dict]:
    """Build a lookup of baseline results keyed by (name, dataset)."""
    if path is None:
        return {}
    return {
        (record["name"], record["dataset"]): record
        for record in load_records(path)
        if "name" in record and "dataset" in record
    }


def to_markdown_table(rows: list[dict]) -> str:
    """Render a compact markdown table."""
    header = [
        "Source",
        "Name",
        "Dataset",
        "Test",
        "Time",
        "Δ test",
        "Speedup",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["source"],
                    row["name"],
                    row["dataset"],
                    row["test"],
                    row["time"],
                    row["delta_test"],
                    row["speedup"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    baseline_map = make_baseline_map(args.baseline)
    rows = []

    for path in args.paths:
        for record in load_records(path):
            name = record.get("name", path.stem)
            dataset = record.get("dataset", "unknown")
            test_metric = record.get("test_metric", record.get("test_accuracy"))
            fit_time = record.get("fit_time")

            delta_test = "—"
            speedup = "—"
            baseline = baseline_map.get((name, dataset))
            if baseline is not None:
                base_test = baseline.get("test_metric", baseline.get("test_accuracy"))
                base_time = baseline.get("fit_time")
                if test_metric is not None and base_test is not None:
                    delta_test = f"{(test_metric - base_test) * 100:+.2f} pts"
                if fit_time and base_time:
                    speedup = f"{base_time / fit_time:.2f}x"

            rows.append(
                {
                    "source": path.name,
                    "name": name,
                    "dataset": dataset,
                    "test": f"{test_metric:.2%}" if test_metric is not None else "—",
                    "time": f"{fit_time:.2f}s" if fit_time is not None else "—",
                    "delta_test": delta_test,
                    "speedup": speedup,
                }
            )

    print(to_markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
