"""Tests for benchmark CLI and result persistence helpers."""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run import (
    BenchmarkResult,
    build_benchmark_parser,
    save_results_csv,
    save_results_json,
    set_benchmark_seed,
)
from benchmarks.run_recurrent import build_recurrent_benchmark_parser


def _sample_result() -> BenchmarkResult:
    return BenchmarkResult(
        name="NEFNet-target_prop",
        dataset="mnist",
        n_neurons=3000,
        activation="abs",
        encoder_strategy="hypersphere",
        solver="tikhonov",
        solver_kwargs={"alpha": 1e-2},
        metric_name="accuracy",
        train_metric=0.99,
        test_metric=0.98,
        fit_time=12.34,
    )


class TestBenchmarkPersistence:
    def test_save_results_json_round_trip(self, tmp_path):
        path = tmp_path / "results.json"
        save_results_json([_sample_result()], path)
        payload = json.loads(path.read_text())
        assert payload[0]["name"] == "NEFNet-target_prop"
        assert payload[0]["solver_kwargs"] == {"alpha": 1e-2}

    def test_save_results_csv_writes_expected_fields(self, tmp_path):
        path = tmp_path / "results.csv"
        save_results_csv([_sample_result()], path)
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["solver_kwargs"] == '{"alpha": 0.01}'


class TestBenchmarkCliParsers:
    def test_set_benchmark_seed_replays_random_streams(self):
        set_benchmark_seed(123)
        first = (random.random(), float(np.random.rand()), float(torch.rand(())))
        set_benchmark_seed(123)
        second = (random.random(), float(np.random.rand()), float(torch.rand(())))
        assert second == first

    def test_feedforward_parser_accepts_save_flags(self, tmp_path):
        parser = build_benchmark_parser()
        args = parser.parse_args(["--save-json", str(tmp_path / "ff.json"), "--multi"])
        assert args.multi
        assert args.save_json.name == "ff.json"
        assert args.seed == 0
        assert args.tp_eta == 0.03
        assert args.tp_e2e_eta == 0.01
        assert not args.tp_project_targets
        assert args.tp_max_infeasible_fraction is None

    def test_recurrent_parser_accepts_save_flags(self, tmp_path):
        parser = build_recurrent_benchmark_parser()
        args = parser.parse_args(
            ["--mode", "pixel", "--save-csv", str(tmp_path / "rec.csv"), "--lstm"]
        )
        assert args.mode == "pixel"
        assert args.lstm
        assert args.save_csv.name == "rec.csv"
        assert args.seed == 0
        assert args.state_target == "reconstruction"
