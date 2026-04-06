"""Predefined Colab experiment suites for GPU-friendly sequential benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmarks.run import format_results, save_results_csv, save_results_json  # noqa: E402
from benchmarks.run_recurrent import run_lstm_baseline, run_streaming_nef  # noqa: E402


def _run_labeled(label: str, fn, /, **kwargs):
    """Run one suite item with explicit progress messages."""
    print(f"Running {label}...", flush=True)
    result = fn(**kwargs)
    print(
        f"Finished {label}: test={result.test_metric:.2%}, fit_time={result.fit_time:.2f}s",
        flush=True,
    )
    return result


def run_row_focus_suite(args: argparse.Namespace) -> list:
    """Run the row-wise sMNIST suite."""
    if args.quick:
        return [
            _run_labeled(
                "StreamNEF row quick",
                run_streaming_nef,
                mode="row",
                n_neurons=512,
                window_size=5,
                alpha=1e-2,
                batch_size=128,
                data_root=args.data_root,
                seed=args.seed,
                solve_mode="woodbury",
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=True,
            ),
            _run_labeled(
                "LSTM row quick",
                run_lstm_baseline,
                mode="row",
                hidden_size=64,
                n_epochs=1,
                batch_size=128,
                data_root=args.data_root,
                seed=args.seed,
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=True,
            ),
        ]

    return [
        _run_labeled(
            "StreamNEF row 2k (woodbury)",
            run_streaming_nef,
            mode="row",
            n_neurons=2000,
            window_size=10,
            alpha=1e-2,
            batch_size=500,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="woodbury",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF row 2k (accumulate)",
            run_streaming_nef,
            mode="row",
            n_neurons=2000,
            window_size=10,
            alpha=1e-2,
            batch_size=500,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF row 8k (woodbury)",
            run_streaming_nef,
            mode="row",
            n_neurons=8000,
            window_size=10,
            alpha=5e-3,
            batch_size=500,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="woodbury",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF row 8k (accumulate)",
            run_streaming_nef,
            mode="row",
            n_neurons=8000,
            window_size=10,
            alpha=5e-3,
            batch_size=500,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "LSTM row 128",
            run_lstm_baseline,
            mode="row",
            hidden_size=128,
            n_epochs=args.lstm_epochs,
            batch_size=args.lstm_batch,
            data_root=args.data_root,
            seed=args.seed,
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
    ]


def run_sequential_hard_suite(args: argparse.Namespace) -> list:
    """Run the longer-sequence sMNIST pixel suites."""
    if args.quick:
        return [
            _run_labeled(
                "StreamNEF pixel quick",
                run_streaming_nef,
                mode="pixel",
                n_neurons=512,
                window_size=28,
                alpha=1e-2,
                batch_size=128,
                data_root=args.data_root,
                seed=args.seed,
                solve_mode="woodbury",
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=True,
            ),
            _run_labeled(
                "LSTM pixel quick",
                run_lstm_baseline,
                mode="pixel",
                hidden_size=64,
                n_epochs=1,
                batch_size=128,
                data_root=args.data_root,
                seed=args.seed,
                device=args.device,
                eval_batch_size=args.eval_batch,
                verbose=True,
            ),
        ]

    return [
        _run_labeled(
            "StreamNEF pixel w28 (accumulate)",
            run_streaming_nef,
            mode="pixel",
            n_neurons=4000,
            window_size=28,
            alpha=1e-2,
            batch_size=250,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF pixel w56 (accumulate)",
            run_streaming_nef,
            mode="pixel",
            n_neurons=4000,
            window_size=56,
            alpha=5e-3,
            batch_size=250,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "StreamNEF permuted w56 (accumulate)",
            run_streaming_nef,
            mode="pixel_permuted",
            n_neurons=4000,
            window_size=56,
            alpha=5e-3,
            batch_size=250,
            data_root=args.data_root,
            seed=args.seed,
            solve_mode="accumulate",
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "LSTM pixel 128",
            run_lstm_baseline,
            mode="pixel",
            hidden_size=128,
            n_epochs=args.lstm_epochs,
            batch_size=args.lstm_batch,
            data_root=args.data_root,
            seed=args.seed,
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
        _run_labeled(
            "LSTM permuted 128",
            run_lstm_baseline,
            mode="pixel_permuted",
            hidden_size=128,
            n_epochs=args.lstm_epochs,
            batch_size=args.lstm_batch,
            data_root=args.data_root,
            seed=args.seed,
            device=args.device,
            eval_batch_size=args.eval_batch,
            verbose=True,
        ),
    ]


SUITES = {
    "row_focus": run_row_focus_suite,
    "sequential_hard": run_sequential_hard_suite,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run predefined Colab benchmark suites")
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./results/colab"))
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=2048)
    parser.add_argument("--lstm-epochs", type=int, default=20)
    parser.add_argument("--lstm-batch", type=int, default=256)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a much smaller suite for validation instead of the full Colab workload",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    results = SUITES[args.suite](args)

    stem = args.suite if not args.quick else f"{args.suite}-quick"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    save_results_json(results, json_path)
    save_results_csv(results, csv_path)

    print()
    print(format_results(results))
    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
