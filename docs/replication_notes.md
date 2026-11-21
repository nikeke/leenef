# B0: frozen teacher zoo (exploratory infrastructure)

`benchmarks.replication.teachers` fits and freezes small CPU classifiers
for later replication work. It contains no distillation, experimental
splits, panels, or evidence of teacher quality.

| `TeacherConfig.kind` | Per-example `input_shape` | `score_kind` |
| --- | --- | --- |
| `mlp1`, `mlp3` | `(d_in,)`; specify `hidden_width` for every hidden layer | `logits` |
| `cnn` | Channels-first `(C, H, W)`, with `H, W >= 2` | `logits` |
| `forest` | `(d_in,)` | `probabilities` |

The MLPs have one or three ReLU hidden layers. The CNN uses padded 3×3
convolutions (8 then 16 channels), ReLU after each convolution, 2×2 max
pooling after the first, and global average pooling before the linear head.

Call `fit_teacher(x_fit, y_fit, model_config, training_config, seed=...)`
with nonempty dense CPU float32 features and int64 labels shaped `(N,)`
in `[0, n_classes)`. There is no implicit conversion, flattening, or
preprocessing. Neural fits use locally seeded initialization and shuffled
mini-batch SGD with cross entropy, fixed learning rate, no dropout, no
validation split, and no early stopping or test-based selection.
`TorchTrainConfig` defaults to one epoch. `ForestTrainConfig` defaults to
eight trees of maximum depth four; scikit-learn fits a bootstrap forest
with Gini splits, `max_features="sqrt"`, explicit `random_state`, and
`n_jobs=1`. Reproducibility assumes the same runtime, CPU execution setup
and input order; it is not promised across library versions or hardware.

Use `predict_scores(x)` for black-box queries. It returns a detached CPU
tensor `(N, n_classes)` with column `j` for class ID `j`. Neural scores
are float32 logits; forest scores are float64 probabilities, using
`classes_` to place observed class columns and zeros for absent classes.
Neural logits still have a column for every prescribed class, even if
that class was absent from fitting. “Scores” means neither that the
values are calibrated nor that probabilities and logits are interchangeable.
`white_box_model()` returns an independent neural snapshot in evaluation
mode with gradients disabled on parameters; changes to a snapshot do not
affect the teacher. Forests expose neither a white-box model nor the
underlying estimator. This API separation is not a confidentiality boundary.

`teacher.save(binary_stream)` and `FrozenTeacher.load(binary_stream)`
round-trip a versioned JSON artifact containing canonical metadata and
named or ordered, little-endian numeric arrays (base64, no pickle).
Forest artifacts carry the split arrays and leaf probabilities for
inference only, not a resumable scikit-learn object. The SHA-256 covers
the architecture, training configuration and seed, class mapping, score
kind, and learned arrays. Loading verifies the digest and validates
the model structure. The embedded digest detects corruption, not malicious
replacement; pass `expected_identity=...` from a trusted record to pin the
contents. Only load suitably sized checkpoints from trusted sources.

When recording actual rows, callers must supply the real
`DatasetBundle.metadata()` (including raw and processed hashes) and
`code_identity` for the experimental source. B0 supplies neither.
Use a clean committed source snapshot: the current supervised
`code_identity` helper hashes untracked path names, not their contents.
Record the teacher identity, fit split/sample budget and labels seen
by the teacher, and separately specify what the replicator can observe
(queries, logits/probabilities/top-1 labels, weights, or true labels).
Report teacher fitting/training cost, including forest conversion, separately
from teacher query time and replication cost. State when the teacher cost is
excluded or amortized, and report the end-to-end wall-clock accounting.
