# leenef method-selection skill

Use this skill when a user wants to choose or configure a leenef method.  The
goal is to map the learning problem to the right fixed-feature analytic-solve
path, not to oversell leenef as a universal replacement for gradient training.

## Decision process

1. Identify the task type:
   - static supervised data -> `NEFLayer`;
   - image supervised data -> `NEFLayer` with receptive-field encoders or
     `ConvNEFPipeline`;
   - sequence classification -> `StreamingNEFClassifier`;
   - class/task streams with stable output semantics -> `partial_fit` /
     `solve_accumulated`;
   - dense discrete-action RL -> `NEFFQIAgent`;
   - sparse-reward RL research -> `NEFSFAgent`.
2. Ask whether the user needs online/continual updates.  If yes, prefer
   accumulation paths over repeated batch `fit`.
3. Ask whether a GPU is available.  If yes, prefer accumulate-and-solve paths
   and chunked statistics for large activity matrices.
4. Check the representation risk.  If fixed features are likely too weak,
   recommend better encoders/front ends before increasing width.

## Conservative defaults

- Static tabular/vector data: `NEFLayer(..., n_neurons=5000, activation="abs",
  encoder_strategy="hypersphere", centers=x_train)`.
- Images: add `encoder_strategy="receptive_field"` and an `image_shape`, or use
  a ConvNEF front end for harder visual data.
- Sequences: `StreamingNEFClassifier(..., window_size=10, n_neurons=8000)`.
- Continual streams: call `partial_fit` for each task/chunk, then
  `solve_accumulated(alpha=(1e-2, "trace"))`.
- Large GPU runs: use `accumulate(..., chunk_stats=True)` followed by `solve`.
- Dense RL: start with non-bootstrapped `target_mode="nstep"` rather than
  one-step TD.
- Sparse RL: start by designing the target or goal representation; do not
  expect width alone to fix sparse rewards.

## Tooling

The installed package exposes:

```bash
leenef-recommend supervised --input-kind image
leenef-recommend temporal --input-kind sequence --gpu
leenef-recommend continual --input-kind image
leenef-recommend rl --gpu
```

The same logic is available in Python:

```python
from leenef.advice import recommend_method

recommendation = recommend_method("supervised", input_kind="image", gpu=True)
```

## Resources

- `docs/USAGE.md` is the user manual.
- `docs/SSUP-plain-language.md` explains the three-paper series without
  assuming machine-learning background.
- `docs/SSUP-I.md`, `docs/SSUP-II.md`, and `docs/SSUP-III.md` are the
  full supervised, continual, and reinforcement-learning manuscripts.
