# Neuron selection — sweep log

Chronological execution log for `research/neuron-selection`.  Synthesis
lives in [`neuron_selection_notes.md`](neuron_selection_notes.md).

## Phase NS-1 — pruning and directed-growth panels

### Pre-registration (written before any confirmatory result exists)

Recorded 2026-11-21, before the harness was run on any dataset other
than the MNIST feasibility probes.  The protocol was reviewed by an
independent model (GPT-6.2 Astra); its must-have changes are
incorporated below.

**Status of the datasets.**

* **MNIST — exploratory.**  Its test set informed the design through the
  probes in `explorations/neuron_selection/`.  MNIST rows are tabulated
  but do not count toward the gate.
* **Fashion-MNIST and CIFAR-10 — confirmatory.**  No test number from
  either dataset has been seen for any method in this line of work.

**Data protocol.**  Flat `div255` vectors.  Per seed, a seeded
permutation of the training set: 10000 validation samples, the rest is
the fit split.  The test set is used for reporting only.  Seeds 0–9.
Every random stream is keyed by (seed, role, generation) and never by
the contrasted factor, so arms are paired by seed and matched arms share
their class draws, centers, and gains.

**Model.**  `NEFLayer`, `abs`, gain $U(0.5, 2)$, data-driven biases from
fit-split samples, one-hot targets.

**Regularization.**  Inside every loop (residuals, rankings, pruning)
the regularizer is the fixed, training-only policy
$\alpha = 10^{-3}$ trace-scaled.  Validation selects $\alpha$ from
$\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ only for the reported decoder
of a finished model and never feeds back.  Both the fixed-$\alpha$ and
the validation-selected metrics are recorded.

**Selection procedure.**  *Batched greedy OBS*: the unit saliency
$s_i = \lVert d_i \rVert^2 / [M^{-1}]_{ii}$ is the exact objective
increase for one deletion at fixed $\lambda$; dropping 5–10 % of the
units per round between recomputations is a greedy approximation, and
nothing here finds the best $k$-subset.

**Pruning panel.**  One 8000-neuron random pool per seed and encoder
family (`hypersphere`, `receptive_field`); nested pools
$P \in \{2000, 4000, 8000\}$; $k \in \{250, 500, 1000, 2000\}$, $k < P$;
criteria `activity`, `magnitude`, `obs_oneshot`, `obs`; plus the
random-width reference curve (leading-$k$ neurons) at widths 250–8000.

**Growth panel.**  Budgets $k \in \{250, 1000\}$, 12 generations, $k$
proposals per generation, prune $2k \to k$.

| Family | Arm | Purpose |
|---|---|---|
| global | `random` | proposal-quality baseline at matched candidates and schedule |
| global | `phd` | the method |
| global | `phd_classshuffle` | residual rows permuted within true class: keeps class-level weights, breaks which image was misfit |
| global | `phd_randkeep` | pHd proposals, uniform random retention from the same $2k$ pool: isolates selection |
| global | `phd_frozen` | pHd bases from generation 0 only: is updating the residual needed? |
| global | `pca` | covariance-shaped proposals: is it just principal-subspace concentration? |
| global | `first_order` | residual-weighted mean directions: tests the "second order is needed" rationale |
| global | `agop` | gradient-outer-product shaping |
| patch | `random` | receptive-field proposals |
| patch | `phd_patch` | pHd on each proposal's patch |

CIFAR-10 runs the first four global arms and both patch arms.

**Primary metric.**  Test accuracy of the model at the final generation
(generation 12), with the decoder $\alpha$ selected on validation.  The
fixed-$\alpha$ test accuracy is reported alongside as a secondary
table.  For the reference rows of the pruning panel the same
validation-selected-$\alpha$ test accuracy is used.

**Primary contrasts** (family `global`, $k = 1000$, paired by seed on
the same split, one-sided paired $t$-tests with
$H_1$: difference $> 0$, Holm-adjusted across the two confirmatory
datasets × three contrasts):

1. `width3x`: `phd` at $k = 1000$ minus an unpruned random layer of
   width 3000 (the leading 3000 neurons of that seed's random pool).
2. `proposal`: `phd` minus `random` proposals.
3. `mechanism`: `phd` minus `phd_classshuffle`.

**Decision rule.**  A contrast *holds* on a dataset iff its
Holm-adjusted one-sided $p$-value is $\le 0.05$.  For `width3x` this
means the selected 1000-unit layer must be significantly *more*
accurate than the random 3000-unit layer; a difference that is merely
indistinguishable from zero does not count.  The unadjusted one-sided
95 % lower confidence bound of each paired difference is reported for
description only.  Seeds missing from either side of a pair are
dropped and the reported $n$ reflects it; a contrast with $n < 8$ is
reported as undecided.

**Gate.**

* *Pass:* contrast 1 and contrast 2 hold on both confirmatory datasets.
  The claim supported is then the narrow one: "at 1000 units, pHd
  proposals with batched greedy OBS selection are more accurate than a
  random layer three times wider".  Proceed with the streaming,
  continual, replication, and compression uses of the loop.
* *Partial:* holds on Fashion-MNIST only.  Proceed with the patch
  variant as the primary form and re-plan before the dependent tracks.
* *Fail:* contrast 2 fails on both.  Stop and re-plan.
* Contrast 3 does not gate; it decides how the mechanism may be
  described.  If it fails, the gain must not be attributed to
  per-sample residual information.

**Secondary, descriptive only:** `selection` contrast (`phd` −
`phd_randkeep`); pruning contrast `obs` − `random` at $P = 4000$,
$k = 1000$; $k = 250$ results; patch family; validation-selected
generation; equivalent random width; generation-1 validation-loss
difference between `phd_classshuffle` and `phd`; wall time and peak
memory.

**Not controlled in this phase** (recorded so they are not forgotten):
activity-scale effects under ridge (columns are not RMS-normalized);
center choice for new neurons (random samples, not high-residual ones);
parameter and operation counts alongside unit counts; comparison with a
gradient-trained MLP of equal width.

**Interpretation caveats fixed in advance.**

* The `first_order` arm proposes only $\pm$ one direction per class
  (20 distinct directions; diversity comes from centers and gains
  alone), so a loss for that arm confounds "first order" with "low
  diversity" and cannot by itself confirm the second-order rationale.
* The top-40 eigenpairs are computed by a block subspace iteration
  converged to $10^{-7}$ relative change in the Ritz values; every
  solve's iteration count and convergence flag is stored with the row,
  and rows containing a non-converged solve are listed by the analyzer.
* The per-dataset summaries are progress reports.  The registered
  analysis is the combined summary over both confirmatory datasets.

**Amendment log.**  2026-11-21, before launch and before any
confirmatory result: after the pre-sweep code review, the primary
metric and the decision rule were made explicit (previously the text
required a lower confidence bound $\ge 0$ for `width3x` without saying
how it combined with the Holm adjustment), and the two caveats above
were added.  No result informed these changes.

**Disclosure.**  2026-11-21, between the amendment above and the
launch: the pipeline was smoke-tested on Fashion-MNIST and CIFAR-10 at
toy sizes (2 seeds, subsampled fit split, 400-neuron pool, widths
$\le 100$, 2 generations), and the head of the resulting Fashion-MNIST
pruning table, including test accuracies, was displayed.  The
pre-sweep code reviewer also ran toy-size Fashion-MNIST smoke rows for
a determinism check and full-size eigensolver comparisons on both
datasets.  None of these are registered cells and no protocol,
analysis, or code change followed from them, but the statement above
that no test number from the confirmatory datasets had been seen holds
only for registered-size configurations.  Smoke tests of confirmatory
pipelines should use the exploratory dataset.

### Execution

**Launch 1 — 2026-11-21T17:47:04Z.**

* Code: commit `7b5b0ae` on `research/neuron-selection`, checked out as
  branch `run/ns-1` in the dedicated worktree `.worktrees/ns1-run`
  (inside the repository directory, the only persistent path visible
  both to the CLI sandbox and to the host).  Pinned identity
  `2d43012f8b9d7644efb9f1267c17bb6c65043f21`.
* Command, from the worktree root:

  ```bash
  PYTHON=/home/amnesia/Persistent/src/leenef/venv/bin/python \
  SEEDS=0-9 THREADS=4 GROWTH_SHARDS=2 \
      bash benchmarks/neuron_selection/run_panels.sh \
      >> results/neuron_selection/logs/driver.log 2>&1 < /dev/null
  ```

* Order: Fashion-MNIST, CIFAR-10, MNIST.  Per dataset: one pruning
  process and two growth shards (seeds 0–4, 5–9), 4 threads each.
* Outputs: `results/neuron_selection/{prune_<dataset>,growth_<dataset>_s<i>}.json`,
  logs and `run_manifest.txt` under `results/neuron_selection/logs/`,
  sentinel `panels_done`, failure marker `panels_failed`.
* Watcher: `benchmarks/neuron_selection/wait_for_panels.sh` (checks
  row counts against the pinned identity).
* Estimated wall time: about 2 h, 8–11 h, and 2 h for the three
  datasets.
* The first launch was made from inside the CLI sandbox, so it dies
  with the CLI session.  The driver resumes from completed rows: to
  continue after an interruption, rerun the same command on the host
  from the worktree root (prefix with `setsid` and append `&`).  A
  second driver refuses to start while one holds the lock.
* Do not commit to branch `run/ns-1` or edit files in that worktree
  while the run is alive.

**Exploratory MNIST precomputation — 2026-11-21.** While the CIFAR-10
workers continue, two separate growth workers may precompute the MNIST
shards (seeds 0–4 and 5–9) using the same pinned code, data, and panel
configuration. Each uses two PyTorch threads rather than the manifest's
four; the workers share a three-CPU affinity mask and run at lower
priority. Their private outputs and logs live under
`results/ns1_precompute/`, outside the active sweep watcher's patterns.

Only a fully completed shard with the expected keys, pinned identity,
seed set, and matching dataset metadata may be published to its normal
output path. Publication must not overwrite an existing file or overlap
the driver's MNIST workers; if that cannot be established, leave the
private output unused. The driver and CIFAR workers are not modified.
The thread count is recorded per row. Different reduction orders may
alter numerical results or discrete selections, and the overlapping
workload makes growth timings and peak memory unsuitable for cross-phase
comparisons. MNIST is exploratory and does not enter the confirmatory
gate. The change was made without examining registered confirmatory
metrics.

**Follow-up — 2026-11-22.** Both private MNIST growth shards completed
and were verified and published before the driver reached MNIST. With
CIFAR growth still active, apply the same private-output and verification
protocol to one two-thread, low-priority MNIST pruning worker on the
same CPU mask, using all ten seeds and the original pruning grid.
The same numerical and timing caveats apply to its rows.

### Results

_To be filled from the analyzer output._
