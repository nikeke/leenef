"""CPU-only frozen classification teachers for exploratory replication."""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass
from typing import BinaryIO, Literal

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import Tensor, nn
from torch.nn import functional as F

from leenef._rng import fork_generator

_SCHEMA = "leenef.replication.teacher.v1"
_DEPTHS = {"mlp1": 1, "mlp3": 3}
_KINDS = (*_DEPTHS, "cnn", "forest")
_F32 = np.dtype("<f4")
_F64 = np.dtype("<f8")
_I64 = np.dtype("<i8")


def _positive_int(value: object, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    """A model architecture and its full input/output contract."""

    kind: Literal["mlp1", "mlp3", "cnn", "forest"]
    input_shape: tuple[int, ...]
    n_classes: int
    hidden_width: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported teacher kind: {self.kind!r}")
        if type(self.input_shape) is not tuple or len(self.input_shape) != (
            3 if self.kind == "cnn" else 1
        ):
            raise ValueError("input_shape must be (C, H, W) for CNNs or (d_in,) otherwise")
        for dim in self.input_shape:
            _positive_int(dim, "input_shape dimension")
        if self.kind == "cnn" and min(self.input_shape[1:]) < 2:
            raise ValueError("CNN inputs require H and W of at least 2")
        _positive_int(self.n_classes, "n_classes")
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if self.kind in _DEPTHS:
            _positive_int(self.hidden_width, "hidden_width")
        elif self.hidden_width is not None:
            raise ValueError("hidden_width is only supported for MLPs")


@dataclass(frozen=True, slots=True)
class TorchTrainConfig:
    """Fixed-epoch mini-batch SGD with cross entropy and a constant LR."""

    epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05

    def __post_init__(self) -> None:
        _positive_int(self.epochs, "epochs")
        _positive_int(self.batch_size, "batch_size")
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        object.__setattr__(self, "lr", float(self.lr))


@dataclass(frozen=True, slots=True)
class ForestTrainConfig:
    """Small bootstrap forest with Gini splits and sklearn defaults otherwise."""

    n_estimators: int = 8
    max_depth: int = 4

    def __post_init__(self) -> None:
        _positive_int(self.n_estimators, "n_estimators")
        _positive_int(self.max_depth, "max_depth")


TrainConfig = TorchTrainConfig | ForestTrainConfig


def _check_training(config: TeacherConfig, training: TrainConfig, seed: int) -> None:
    if not isinstance(config, TeacherConfig):
        raise TypeError("config must be a TeacherConfig")
    expected = ForestTrainConfig if config.kind == "forest" else TorchTrainConfig
    if not isinstance(training, expected):
        raise TypeError(f"{config.kind} requires {expected.__name__}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def _check_tensor(x: Tensor, name: str, dtype: torch.dtype) -> None:
    if (
        not isinstance(x, Tensor)
        or x.device.type != "cpu"
        or x.layout != torch.strided
        or x.dtype != dtype
    ):
        raise TypeError(f"{name} must be a dense CPU {dtype} tensor")


def _features(x: Tensor, config: TeacherConfig) -> Tensor:
    _check_tensor(x, "x", torch.float32)
    if x.ndim != len(config.input_shape) + 1 or x.shape[1:] != config.input_shape or not len(x):
        raise ValueError(
            f"x must have nonempty batch dimension and sample shape {config.input_shape}"
        )
    if not x.isfinite().all():
        raise ValueError("x must contain only finite values")
    return x.detach()


def _labels(y: Tensor, n_samples: int, n_classes: int) -> Tensor:
    _check_tensor(y, "y", torch.int64)
    if y.shape != (n_samples,):
        raise ValueError("y must have shape (N,) matching x")
    if ((y < 0) | (y >= n_classes)).any():
        raise ValueError("y must contain integer class IDs in [0, n_classes)")
    return y


def _torch_model(config: TeacherConfig) -> nn.Sequential:
    # Meta tensors let the standard module constructors run without consuming CPU RNG.
    opts = {"device": "meta", "dtype": torch.float32}
    if config.kind == "cnn":
        return nn.Sequential(
            nn.Conv2d(config.input_shape[0], 8, 3, padding=1, **opts),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, **opts),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(16, config.n_classes, **opts),
        )
    width = config.hidden_width
    assert width is not None
    layers: list[nn.Module] = []
    d_in = config.input_shape[0]
    for _ in range(_DEPTHS[config.kind]):
        layers.extend((nn.Linear(d_in, width, **opts), nn.ReLU()))
        d_in = width
    layers.append(nn.Linear(d_in, config.n_classes, **opts))
    return nn.Sequential(*layers)


@torch.inference_mode(False)
def _train_torch(
    x: Tensor, y: Tensor, config: TeacherConfig, training: TorchTrainConfig, seed: int
) -> nn.Sequential:
    root = torch.Generator(device="cpu").manual_seed(seed)
    init_rng, shuffle_rng = fork_generator(root), fork_generator(root)
    model = _torch_model(config).to_empty(device="cpu")
    with torch.no_grad():
        for layer in model:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                bound = 1 / math.sqrt(math.prod(layer.weight.shape[1:]))
                layer.weight.uniform_(-bound, bound, generator=init_rng)
                layer.bias.uniform_(-bound, bound, generator=init_rng)
    optimizer = torch.optim.SGD(model.parameters(), lr=training.lr, foreach=False)
    with torch.enable_grad(), torch.autocast("cpu", enabled=False):
        for _ in range(training.epochs):
            order = torch.randperm(len(x), generator=shuffle_rng, device="cpu")
            for batch in order.split(training.batch_size):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(x[batch]), y[batch]).backward()
                optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not p.isfinite().all() for p in model.parameters()):
        raise ValueError("training produced nonfinite weights")
    return model.eval().requires_grad_(False)


@dataclass(frozen=True, slots=True)
class _Tree:
    children: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    probabilities: np.ndarray

    @classmethod
    def extract(cls, estimator, x: np.ndarray) -> _Tree:
        tree = estimator.tree_
        # Read actual predict_proba values: sklearn releases differ in whether
        # tree_.value stores counts or proportions. Every leaf has a fit example.
        leaves, first_rows = np.unique(estimator.apply(x), return_index=True)
        probabilities = estimator.predict_proba(x[first_rows])
        table = np.zeros((tree.node_count, probabilities.shape[1]), dtype=_F64)
        table[leaves] = probabilities
        return cls(
            np.column_stack((tree.children_left, tree.children_right)),
            tree.feature.copy(),
            tree.threshold.copy(),
            table,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        nodes = np.zeros(len(x), dtype=np.intp)
        rows = np.arange(len(x))
        while rows.size:
            current = nodes[rows]
            branches = self.children[current, 0] != -1
            rows, current = rows[branches], current[branches]
            right = x[rows, self.features[current]] > self.thresholds[current]
            nodes[rows] = self.children[current, right.astype(np.intp)]
        return self.probabilities[nodes]

    def validate(self, n_features: int, n_present: int) -> None:
        n = len(self.features)
        if (
            n == 0
            or self.features.shape != (n,)
            or self.children.shape != (n, 2)
            or self.thresholds.shape != (n,)
            or self.probabilities.shape != (n, n_present)
        ):
            raise ValueError("invalid tree array shapes")
        leaf = self.children[:, 0] == -1
        branch = ~leaf
        children = self.children[branch]
        if (
            (self.children[leaf] != -1).any()
            or (children <= np.flatnonzero(branch)[:, None]).any()
            or (children >= n).any()
            or not np.array_equal(np.sort(children.ravel()), np.arange(1, n))
            or ((self.features[branch] < 0) | (self.features[branch] >= n_features)).any()
            or (self.features[leaf] != -2).any()
            or not np.isfinite(self.thresholds).all()
            or (self.probabilities[branch] != 0).any()
            or not np.isfinite(self.probabilities).all()
            or (self.probabilities < 0).any()
            or not np.allclose(self.probabilities[leaf].sum(axis=1), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("invalid tree structure or probabilities")


def _train_forest(
    x: Tensor, y: Tensor, training: ForestTrainConfig, seed: int
) -> tuple[tuple[int, ...], tuple[_Tree, ...]]:
    values = x.numpy()
    forest = RandomForestClassifier(
        **asdict(training),
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    ).fit(values, y.numpy())
    classes = tuple(forest.classes_.tolist())
    trees = tuple(_Tree.extract(estimator, values) for estimator in forest.estimators_)
    for tree in trees:
        tree.validate(values.shape[1], len(classes))
    return classes, trees


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate checkpoint key")
    return obj


def _encode_array(array: np.ndarray, dtype: np.dtype) -> dict:
    data = array.astype(dtype, copy=False).tobytes(order="C")
    return {
        "shape": list(array.shape),
        "dtype": dtype.str,
        "data": base64.b64encode(data).decode("ascii"),
    }


def _decode_array(obj: dict, dtype: np.dtype, ndim: int) -> np.ndarray:
    if (
        not isinstance(obj, dict)
        or set(obj) != {"dtype", "shape", "data"}
        or obj["dtype"] != dtype.str
        or not isinstance(obj["shape"], list)
        or len(obj["shape"]) != ndim
        or any(type(dim) is not int or dim < 1 for dim in obj["shape"])
        or not isinstance(obj["data"], str)
    ):
        raise ValueError("invalid checkpoint array descriptor")
    data = base64.b64decode(obj["data"], validate=True)
    if len(data) != math.prod(obj["shape"]) * dtype.itemsize:
        raise ValueError("checkpoint array byte count does not match shape")
    if base64.b64encode(data).decode("ascii") != obj["data"]:
        raise ValueError("checkpoint array data is not canonical base64")
    return np.frombuffer(data, dtype=dtype).reshape(obj["shape"])


class FrozenTeacher:
    """Inference-only teacher. Construct using :func:`fit_teacher` or :meth:`load`."""

    def __init__(
        self,
        config: TeacherConfig,
        training: TrainConfig,
        seed: int,
        class_ids: tuple[int, ...],
        learned: nn.Sequential | tuple[_Tree, ...],
    ) -> None:
        self._config = config
        self._training = training
        self._seed = seed
        self._class_ids = class_ids
        self._learned = learned

    @property
    def score_kind(self) -> Literal["logits", "probabilities"]:
        return "probabilities" if self._config.kind == "forest" else "logits"

    def predict_scores(self, x: Tensor) -> Tensor:
        """Return detached CPU scores of shape ``(N, n_classes)`` in class-ID order."""
        x = _features(x, self._config)
        if isinstance(self._learned, nn.Sequential):
            with torch.no_grad(), torch.autocast("cpu", enabled=False):
                scores = self._learned(x)
            if not scores.isfinite().all():
                raise ValueError("teacher produced nonfinite logits")
            return scores
        values = x.numpy()
        present = np.zeros((len(x), len(self._class_ids)), dtype=np.float64)
        for tree in self._learned:
            present += tree.predict(values)
        scores = np.zeros((len(x), self._config.n_classes), dtype=np.float64)
        scores[:, self._class_ids] = present / len(self._learned)
        return torch.from_numpy(scores)

    def white_box_model(self) -> nn.Sequential:
        """Return an independent frozen torch model, never the teacher's live state."""
        if not isinstance(self._learned, nn.Sequential):
            raise NotImplementedError("the forest teacher is black-box only")
        return copy.deepcopy(self._learned)

    @property
    def metadata(self) -> dict:
        """Fresh JSON-compatible metadata, with no inferred dataset provenance."""
        return {
            "model": {**asdict(self._config), "input_shape": list(self._config.input_shape)},
            "trainer": (
                "sklearn_random_forest" if self._config.kind == "forest" else "sgd_cross_entropy"
            ),
            "training": asdict(self._training),
            "seed": self._seed,
            "score_kind": self.score_kind,
            # Source columns: forest uses sklearn.classes_; neural logits have all columns.
            "class_ids": list(self._class_ids),
        }

    def _payload(self) -> dict:
        if isinstance(self._learned, nn.Sequential):
            arrays = {
                name: _encode_array(tensor.numpy(), _F32)
                for name, tensor in self._learned.state_dict().items()
            }
        else:
            arrays = [
                {
                    "children": _encode_array(tree.children, _I64),
                    "features": _encode_array(tree.features, _I64),
                    "thresholds": _encode_array(tree.thresholds, _F64),
                    "probabilities": _encode_array(tree.probabilities, _F64),
                }
                for tree in self._learned
            ]
        return {"schema": _SCHEMA, "metadata": self.metadata, "arrays": arrays}

    @property
    def identity(self) -> str:
        """SHA-256 of canonical metadata and ordered, typed learned arrays."""
        return _identity(self._payload())

    def save(self, stream: BinaryIO) -> None:
        """Write the checkpoint to an open binary stream."""
        payload = self._payload()
        stream.write(_canonical({"sha256": _identity(payload), "payload": payload}))

    @classmethod
    @torch.inference_mode(False)
    def load(cls, stream: BinaryIO, *, expected_identity: str | None = None) -> FrozenTeacher:
        """Verify and load a numeric checkpoint; optionally pin a trusted SHA-256."""
        try:
            envelope = json.loads(stream.read(), object_pairs_hook=_unique_object)
            if not isinstance(envelope, dict) or set(envelope) != {"sha256", "payload"}:
                raise ValueError("invalid checkpoint envelope")
            payload, digest = envelope["payload"], envelope["sha256"]
            actual = _identity(payload)
            if not isinstance(digest, str) or not hmac.compare_digest(actual, digest):
                raise ValueError("checkpoint SHA-256 mismatch")
            if expected_identity is not None and actual != expected_identity:
                raise ValueError("checkpoint does not match expected identity")
            return cls._restore(payload)
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ValueError(f"invalid teacher checkpoint: {error}") from error

    @classmethod
    def _restore(cls, payload: dict) -> FrozenTeacher:
        if set(payload) != {"schema", "metadata", "arrays"} or payload["schema"] != _SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        meta, arrays = payload["metadata"], payload["arrays"]
        model_cfg = {**meta["model"], "input_shape": tuple(meta["model"]["input_shape"])}
        config = TeacherConfig(**model_cfg)
        forest = config.kind == "forest"
        training = (ForestTrainConfig if forest else TorchTrainConfig)(**meta["training"])
        seed = meta["seed"]
        _check_training(config, training, seed)
        class_ids = tuple(meta["class_ids"])
        if (
            not class_ids
            or any(type(c) is not int or not 0 <= c < config.n_classes for c in class_ids)
            or tuple(sorted(set(class_ids))) != class_ids
            or (not forest and len(class_ids) != config.n_classes)
        ):
            raise ValueError("invalid checkpoint class mapping")
        if forest:
            if not isinstance(arrays, list) or len(arrays) != training.n_estimators:
                raise ValueError("invalid checkpoint forest size")
            trees = []
            for obj in arrays:
                if set(obj) != {"children", "features", "thresholds", "probabilities"}:
                    raise ValueError("invalid checkpoint tree fields")
                tree = _Tree(
                    children=_decode_array(obj["children"], _I64, 2),
                    features=_decode_array(obj["features"], _I64, 1),
                    thresholds=_decode_array(obj["thresholds"], _F64, 1),
                    probabilities=_decode_array(obj["probabilities"], _F64, 2),
                )
                tree.validate(config.input_shape[0], len(class_ids))
                trees.append(tree)
            learned = tuple(trees)
        else:
            model = _torch_model(config)
            state = model.state_dict()
            if not isinstance(arrays, dict) or state.keys() != arrays.keys():
                raise ValueError("invalid checkpoint weight names")
            weights = {}
            for name, tensor in state.items():
                array = _decode_array(arrays[name], _F32, tensor.ndim)
                if array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
                    raise ValueError(f"invalid checkpoint weight: {name}")
                weights[name] = torch.from_numpy(array.astype(np.float32, copy=True))
            model.to_empty(device="cpu").load_state_dict(weights)
            learned = model.eval().requires_grad_(False)
        teacher = cls(config, training, seed, class_ids, learned)
        if _canonical(teacher.metadata) != _canonical(meta):
            raise ValueError("invalid checkpoint metadata")
        return teacher


def fit_teacher(
    x: Tensor,
    y: Tensor,
    config: TeacherConfig,
    training: TrainConfig,
    *,
    seed: int,
) -> FrozenTeacher:
    """Train on caller-supplied fit tensors only and return an inference-only teacher."""
    _check_training(config, training, seed)
    x = _features(x, config)
    y = _labels(y, len(x), config.n_classes)
    if isinstance(training, ForestTrainConfig):
        class_ids, learned = _train_forest(x, y, training, seed)
    else:
        class_ids = tuple(range(config.n_classes))
        learned = _train_torch(x, y, config, training, seed)
    return FrozenTeacher(config, training, seed, class_ids, learned)
