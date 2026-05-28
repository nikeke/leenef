# leenef usage manual

leenef is a PyTorch package for fixed-feature NEF-style models with analytic
readouts.  It is useful when you want a compact, auditable learner whose state
is the pair of sufficient statistics $A^\top A$ and $A^\top Y$.

## Installation

```bash
pip install leenef
```

For development from a checkout:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

The core package depends on PyTorch, torchvision, and scikit-learn.  Benchmark
and paper-generation dependencies are optional and are not needed for ordinary
library use.

## Choosing a method

Use the `leenef-recommend` helper for a conservative starting point:

```bash
leenef-recommend supervised --input-kind image
leenef-recommend temporal --input-kind sequence --gpu
leenef-recommend continual --input-kind image
leenef-recommend rl --gpu
```

The main choices are:

| Situation | Start with | Why |
|---|---|---|
| Static classification or regression | `NEFLayer` | fixed encoders plus one analytic decoder solve |
| Image data with local structure | `NEFLayer(..., encoder_strategy="receptive_field")` or `ConvNEFPipeline` | gives the fixed features spatial locality |
| Ensembles | `NEFEnsemble` | averages independent feature banks |
| Sequence classification | `StreamingNEFClassifier` | delay-line windows plus pooled analytic readout |
| Continual class streams | `NEFLayer.partial_fit` + `solve_accumulated` | additive sufficient statistics match the joint fixed-feature solution |
| Dense discrete-action RL | `NEFFQIAgent` | per-action analytic value regression |
| Sparse RL research | `NEFSFAgent` | successor-feature targets expose sparse reward structure |

## Supervised learning

```python
import torch
from leenef import NEFLayer

x_train = torch.randn(1000, 20)
y_index = (x_train[:, 0] > 0).long()
y_train = torch.zeros(1000, 2)
y_train[torch.arange(1000), y_index] = 1.0

layer = NEFLayer(
    d_in=20,
    n_neurons=1000,
    d_out=2,
    activation="abs",
    encoder_strategy="hypersphere",
    centers=x_train,
)
layer.fit(x_train, y_train, alpha=(1e-2, "trace"))
pred = layer(torch.randn(8, 20))
```

Use `centers=x_train` when you can: it places the absolute-value folds on the
data manifold instead of at generic random offsets.  For image vectors, try
`encoder_strategy="receptive_field"` with `encoder_kwargs={"image_shape": ...}`.

## Incremental and continual learning

When data arrives in chunks, accumulate normal-equation statistics and solve
once:

```python
online = NEFLayer(20, 1000, 2, centers=x_train)
for batch_x, batch_y in stream:
    online.partial_fit(batch_x, batch_y)
online.solve_accumulated(alpha=(1e-2, "trace"))
```

This is the core continual-learning identity: if the output columns keep the
same meaning, sequential accumulation gives the same fixed-feature readout as a
joint solve over the union of the chunks.

## Sequence classification

```python
from leenef import StreamingNEFClassifier

clf = StreamingNEFClassifier(
    d_timestep=1,
    n_neurons=8000,
    d_out=10,
    window_size=10,
    centers=x_train_seq,
)
clf.fit(x_train_seq, y_train)
```

For large CPU or GPU runs, prefer:

```python
for batch_x, batch_y in stream:
    clf.accumulate(batch_x, batch_y, chunk_stats=True)
clf.solve(alpha=(1e-2, "trace"))
```

The chunked accumulate path avoids materializing the full activity matrix and is
the preferred production path for large models.

## GPU use

All `torch.nn.Module` components move with ordinary PyTorch calls:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
layer = NEFLayer(20, 1000, 2).to(device)
layer.fit(x_train.to(device), y_train.to(device))
```

For `StreamingNEFClassifier`, use `accumulate(..., chunk_stats=True)` followed
by `solve()` on GPU-scale runs.  RL components accept a `device=` argument:

```python
from leenef import NEFFQIAgent, NEFSFAgent

agent = NEFFQIAgent(d_obs=8, n_actions=4, n_neurons=2000, device=device)
sf_agent = NEFSFAgent(d_obs=8, n_actions=4, n_neurons=2000, device=device)
```

## Reinforcement learning caveat

The RL code is a research component for analytic target construction.  It is
not a drop-in replacement for tuned RL libraries.  Start with stable
non-bootstrapped targets for dense control.  Sparse-reward tasks usually need a
goal, hindsight, or successor-feature target that makes $A^\top Y$ informative.

## Reproducibility

Pass an explicit `torch.Generator` or seed whenever possible.  The benchmark
harnesses in `benchmarks/` add code identities, dataset hashes, and per-seed
result rows for paper-grade experiments; ordinary library users do not need
that machinery unless they are reproducing the SSUP papers.
