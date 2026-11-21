"""Small, synthetic contract tests for the frozen replication teachers."""

import base64
import hashlib
import io
import json
import struct

import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn

from benchmarks.replication.teachers import (
    ForestTrainConfig,
    FrozenTeacher,
    TeacherConfig,
    TorchTrainConfig,
    fit_teacher,
)


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(params=["mlp1", "mlp3", "cnn", "forest"])
def tiny_problem(request):
    kind = request.param
    shape = (1, 4, 5) if kind == "cnn" else (3,)
    config = TeacherConfig(kind, shape, 4, hidden_width=5 if kind.startswith("mlp") else None)
    training = (
        ForestTrainConfig(n_estimators=5, max_depth=3)
        if kind == "forest"
        else TorchTrainConfig(epochs=2, batch_size=3, lr=0.03)
    )
    rng = torch.Generator(device="cpu").manual_seed(28)
    x = torch.randn(9, *shape, generator=rng)
    y = torch.tensor([3, 1, 3, 1, 1, 3, 3, 1, 3])
    return x, y, config, training


def _checkpoint(teacher):
    stream = io.BytesIO()
    teacher.save(stream)
    stream.seek(0)
    return stream


def _edited_checkpoint(envelope, *, rehash=False):
    if rehash:
        canonical = json.dumps(
            envelope["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        envelope["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return io.BytesIO(json.dumps(envelope).encode())


def test_reproducible_frozen_round_trip_and_tamper(tiny_problem):
    x, y, config, training = tiny_problem
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(333)
        before = torch.get_rng_state()
        first = fit_teacher(x, y, config, training, seed=12)
        assert torch.equal(torch.get_rng_state(), before)

        torch.random.default_generator.manual_seed(555)
        before = torch.get_rng_state()
        with torch.inference_mode():
            second = fit_teacher(x, y, config, training, seed=12)
            restored = FrozenTeacher.load(_checkpoint(first), expected_identity=first.identity)
        assert torch.equal(torch.get_rng_state(), before)

    query = x.flip(0).requires_grad_()
    scores = first.predict_scores(query)
    assert scores.shape == (len(x), config.n_classes) and scores.device.type == "cpu"
    assert not scores.requires_grad and scores.grad_fn is None
    assert torch.equal(scores, second.predict_scores(query))
    assert torch.equal(scores, restored.predict_scores(query))
    assert first.identity == second.identity == restored.identity
    assert first.metadata == restored.metadata
    assert first.metadata["seed"] == 12
    assert first.metadata["model"]["input_shape"] == list(config.input_shape)

    if config.kind == "forest":
        assert first.score_kind == "probabilities"
        with pytest.raises(NotImplementedError, match="black-box only"):
            restored.white_box_model()
    else:
        assert first.score_kind == "logits"
        snapshot = restored.white_box_model()
        assert all(not module.training for module in snapshot.modules())
        assert all(not p.requires_grad and p.grad is None for p in snapshot.parameters())
        assert torch.equal(snapshot(query.detach()), scores)
        assert (
            sum(isinstance(layer, nn.ReLU) for layer in snapshot)
            == {
                "mlp1": 1,
                "mlp3": 3,
                "cnn": 2,
            }[config.kind]
        )
        snapshot.train()
        with torch.no_grad():
            list(snapshot.parameters())[-1].add_(10)
        assert torch.equal(restored.predict_scores(query), scores)
        assert not restored.white_box_model().training

    envelope = json.load(_checkpoint(first))
    arrays = envelope["payload"]["arrays"]
    descriptor = (
        arrays[0]["thresholds"] if config.kind == "forest" else next(iter(arrays.values()))
    )
    data = bytearray(base64.b64decode(descriptor["data"]))
    fmt = "<d" if config.kind == "forest" else "<f"
    struct.pack_into(fmt, data, 0, struct.unpack_from(fmt, data)[0] + 1)
    descriptor["data"] = base64.b64encode(data).decode()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FrozenTeacher.load(_edited_checkpoint(envelope))
    changed = FrozenTeacher.load(_edited_checkpoint(envelope, rehash=True))
    assert changed.identity != first.identity
    with pytest.raises(ValueError, match="expected identity"):
        FrozenTeacher.load(_edited_checkpoint(envelope), expected_identity=first.identity)


def test_forest_probabilities_match_sklearn_with_missing_classes():
    x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([3, 1, 3, 3, 1, 1])
    cfg = TeacherConfig("forest", (1,), 4)
    training = ForestTrainConfig(n_estimators=7, max_depth=2)
    query = torch.linspace(-3, 4, 29).reshape(-1, 1)
    teacher = fit_teacher(x, y, cfg, training, seed=27)
    expected = RandomForestClassifier(n_estimators=7, max_depth=2, random_state=27, n_jobs=1).fit(
        x.numpy(), y.numpy()
    )
    assert teacher.metadata["class_ids"] == expected.classes_.tolist() == [1, 3]
    scores = teacher.predict_scores(query)
    assert torch.equal(scores[:, [1, 3]], torch.from_numpy(expected.predict_proba(query.numpy())))
    assert torch.count_nonzero(scores[:, [0, 2]]) == 0

    remapped = fit_teacher(x, y - 1, cfg, training, seed=27)
    assert remapped.identity != teacher.identity
    assert torch.equal(remapped.predict_scores(query)[:, [0, 2]], scores[:, [1, 3]])
    # Only the declared output width changes; fitted forest arrays and seed are unchanged.
    wider = fit_teacher(x, y, TeacherConfig("forest", (1,), 5), training, seed=27)
    assert wider.identity != teacher.identity
    assert torch.equal(wider.predict_scores(query)[:, :4], scores)


def test_forest_rejects_invalid_structure_even_with_recomputed_digest():
    x = torch.tensor([[0.0], [1.0]])
    teacher = fit_teacher(
        x, torch.tensor([1, 1]), TeacherConfig("forest", (1,), 2), ForestTrainConfig(), seed=2
    )
    assert torch.equal(teacher.predict_scores(x), torch.tensor([[0.0, 1.0]] * 2))
    envelope = json.load(_checkpoint(teacher))
    children = envelope["payload"]["arrays"][0]["children"]
    children["data"] = base64.b64encode(struct.pack("<qq", 0, 0)).decode()
    with pytest.raises(ValueError, match="invalid tree structure"):
        FrozenTeacher.load(_edited_checkpoint(envelope, rehash=True))


def test_invalid_inputs_rejected_for_fit_and_predict(tiny_problem):
    x, y, config, training = tiny_problem
    teacher = fit_teacher(x, y, config, training, seed=3)
    for invalid in (None, x.to(torch.float64), x[0], x[:0], x * torch.nan):
        with pytest.raises((TypeError, ValueError), match=r"\bx\b"):
            fit_teacher(invalid, y, config, training, seed=3)
        with pytest.raises((TypeError, ValueError), match=r"\bx\b"):
            teacher.predict_scores(invalid)
    for invalid in (None, y.float(), y[:, None], y - 4, y + 4, y[:-1]):
        with pytest.raises((TypeError, ValueError), match=r"\by\b"):
            fit_teacher(x, invalid, config, training, seed=3)


@pytest.mark.parametrize(
    "kind,shape,classes,width",
    [
        ("unknown", (3,), 3, 4),
        ("cnn", (1, 1, 4), 3, None),
        ("cnn", (12,), 3, None),
        ("forest", (3,), 3, 4),
        ("mlp3", (3,), 3, None),
        ("mlp1", (3,), 1, 4),
        ("mlp1", (True,), 3, 4),
    ],
)
def test_invalid_model_config(kind, shape, classes, width):
    with pytest.raises(ValueError):
        TeacherConfig(kind, shape, classes, width)


def test_invalid_training_config_and_seed():
    x, y, config = (
        torch.ones(1, 2),
        torch.zeros(1, dtype=torch.long),
        TeacherConfig("mlp1", (2,), 2, 3),
    )
    with pytest.raises(TypeError, match="requires TorchTrainConfig"):
        fit_teacher(x, y, config, ForestTrainConfig(), seed=0)
    with pytest.raises(ValueError, match="seed"):
        fit_teacher(x, y, config, TorchTrainConfig(), seed=-1)
    with pytest.raises(ValueError, match="lr"):
        TorchTrainConfig(lr=float("nan"))
