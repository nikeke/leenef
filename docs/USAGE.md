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

## Choosing and configuring a method

The first decision is whether your task is static, sequential, continual, or
control-oriented.  The second is whether the fixed feature map is likely to
contain the structure you need.  If not, prefer better encoders or a front end
before simply increasing neuron count.

| Situation | Start with | Conservative configuration |
|---|---|---|
| Static vector/tabular classification or regression | `NEFLayer` | `n_neurons=1000-5000`, `activation="abs"`, `encoder_strategy="hypersphere"`, `gain=(0.5, 2.0)`, `centers=x_train`, `alpha=(1e-3, "trace")` to match the SSUP static panels or `alpha=(1e-2, "trace")` as the API default |
| Static image vectors | `NEFLayer` with receptive-field encoders | `n_neurons=5000`, `encoder_strategy="receptive_field"`, `encoder_kwargs={"image_shape": (H, W)}`, data-driven centers, trace-scaled alpha in the `1e-3` to `1e-2` range |
| Harder visual tasks | `ConvNEFPipeline` | parallel PCA/conv stages, feature standardization, and a NEF head; use this before treating generic random encoders as a serious visual baseline |
| Ensembles | `NEFEnsemble` | 5-20 members when wall-clock permits; use independent seeds and probability averaging unless you specifically need voting |
| Sequence classification | `StreamingNEFClassifier` | `window_size=10`, `n_neurons=4000-8000`, data-driven centers, `alpha=(1e-4, "trace")`, and `accumulate(..., chunk_stats=True)` followed by `solve()` for large runs |
| Continual class/task streams | `NEFLayer.partial_fit` + `solve_accumulated` | keep output columns stable, use `n_neurons=5000`, `centers_mode="first_task"` or representative centers, `alpha=(1e-3, "trace")`, and solve once after accumulating all task statistics |
| Streaming continual sequences | `StreamingNEFClassifier.accumulate` | `window_size=10`, `n_neurons=4000+`, `alpha=(1e-4, "trace")`, chunked statistics, and GPU if available |
| Dense discrete-action RL | `NEFAVRAgent` | research default: `target_mode="nstep"`, `n_step=100`, `n_neurons=6000`, `gamma=0.98`, `alpha=0.1`, `forget_factor=0.999`, `solve_every=10`; avoid one-step TD as the first target |
| Sparse-reward RL research | `NEFSFAgent` | start with `n_neurons=2000`, task-matched successor or directional readout, explicit goal/hindsight structure, and `device="cuda"` for full-successor matrices when available |

Alpha policy matters.  Normal-equation solves should use explicit
`(value, "trace")` policies so the diagonal regularizer scales with the average
activity energy.  Woodbury/update-inverse paths use fixed alpha because the
cached inverse is parameterized by that exact diagonal value; do not change
`fixed_alpha` mid-stream without `refresh_inverse()` or `reset_continuous()`.

The root [`SKILL.md`](../SKILL.md) is a standard Agent Skill with YAML
frontmatter and the same decision logic.  It references reference material
(this manual and the SSUP papers) via relative `docs/` paths so the agent can
read them on demand.  Use it when you want an agent to inspect your data
shape, task stream, hardware, and accuracy/speed constraints before proposing
a configuration.

### Installing the skill

The skill plus its reference docs ship inside the installed package, so a
plain `pip install leenef` is enough to obtain them.  Materialize a portable,
self-contained skill directory with the bundled installer:

```bash
# Copy the skill (SKILL.md + docs/) into a destination skill directory.
leenef-install-skill --dest ~/.config/agent/skills
# or symlink instead of copy (development checkouts only):
leenef-install-skill --dest ~/.config/agent/skills --link
```

This writes `<dest>/leenef-method-selection/SKILL.md` next to mirrored
`docs/`, `src/`, `benchmarks/`, `tests/`, `colab/`, and `results/` folders
that contain this manual, the three SSUP papers, the project notes, the
package source, benchmark drivers, test suite, Colab notebooks, and the
`base_config*.json` defaults — so the skill's relative reference links and
the agent's spot-checks of actual APIs or configs all resolve wherever it is
installed.  From a repository checkout you can equivalently use `--link` (or
copy the root `SKILL.md` together with those folders by hand).

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
layer.fit(x_train, y_train, alpha=(1e-3, "trace"))
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
from leenef import NEFAVRAgent, NEFSFAgent

agent = NEFAVRAgent(d_obs=8, n_actions=4, n_neurons=2000, device=device)
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
