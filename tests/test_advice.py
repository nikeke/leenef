"""Tests for leenef method-selection helpers."""

from __future__ import annotations

import pytest

from leenef.advice import main, recommend_method


def test_recommend_supervised_image_uses_spatial_encoder():
    rec = recommend_method("supervised", input_kind="image", gpu=True)

    assert rec.method == "NEFLayer"
    assert rec.starting_config["encoder_strategy"] == "receptive_field"
    assert rec.starting_config["fit_path"] == "accumulate + solve"


def test_recommend_temporal_uses_streaming_classifier():
    rec = recommend_method("temporal", input_kind="sequence")

    assert rec.method == "Stream-NEF"
    assert rec.starting_config["class"] == "StreamingNEFClassifier"


def test_recommend_rejects_unknown_task():
    with pytest.raises(ValueError, match="task must be one of"):
        recommend_method("unknown")


def test_cli_prints_recommendation(capsys):
    main(["continual", "--input-kind", "image"])

    out = capsys.readouterr().out
    assert "Method:" in out
    assert "partial_fit" in out
