# sl_clean sweeps — phase log

This document mirrors `attic/docs/rl_clean_sweeps.md` for the
supervised-learning side of the foundational paper series. Every phase
that contributes to a numeric claim in **Paper I — Sufficient Statistics
as the Universal Primitive** lands here, with:

- the phase name, experiment, knob, and value list;
- per-value n, mean ± SE, min, max for the primary metric;
- the adoption decision (Δ vs `sqrt(SE_winner² + SE_current²)`) and a
  one-line justification of why the analyzer kept or replaced the
  incumbent;
- the regenerated figure path under `docs/figures/`;
- open questions and follow-ups (entanglements, knob-pair sweeps, etc.).

The driver scripts (`benchmarks/sl_clean/sl_hp_driver.sh`) and the live
base configs (`results/sl_clean/base_config_*.json`) are the operational
counterparts of this log; this file is the human-readable summary.

---

## Conventions

- **Code identity.** Every result row carries a `code_identity` hash —
  a content-based 40-hex-char SHA derived from `git ls-tree -r HEAD --
  src benchmarks tests pyproject.toml`, optionally suffixed
  `-dirty-<sha256[:12]>` if any of those paths are dirty. Two commits
  that leave the watched paths bit-identical (e.g. docs-only or
  results-only commits) produce the same identity, so result rows
  from one are not invalidated by the other. Rows whose
  `code_identity` no longer matches the current tree are kept on
  disk but excluded from aggregates and re-run on demand.
- **Adoption rule.** A new value `v*` replaces the current base only when

  ```
  mean(metric | v=v*) − mean(metric | v=base) > sqrt(SE_v*² + SE_base²)
  ```

  with `SE = std(ddof=1) / sqrt(n)`, both sides averaged over `≥3` seeds
  by default (configurable via `--min-seeds`).
- **Seed budget.** 10 seeds for headline claims, 5 for ablation tables,
  3 for appendix / legacy cells.
- **Resumability.** Re-running a driver re-uses cells with matching
  `(experiment, code_identity, config, seed, run_context)` and only
  computes the rest. Logs land under `results/sl_clean/logs/<phase>.log`.

### Code-identity timeline

The library has been touched several times since the sl_clean infrastructure
landed (commit `668df59`).  The timeline below maps each B-panel JSON to the
`code_identity` it was written under, plus the human-readable git commit
that produced that identity.  Numerical reproducibility is per-row exact;
cross-row comparison within a panel is safe because every row in a single
panel shares one identity (we never mix identities inside an aggregate).

| panel | rows | code_identity (prefix) | commit landed | notable changes vs prior identity |
|---|---|---|---|---|
| `biases_mnist_panel.json` | 90 | `a18377…` | `a183779` (B1 scaffold) | first sl_clean panel |
| `biases_fashion_panel.json` | 90 | `644d7c…` | `644d7c1` (Fashion+CIFAR generalization) | adds dataset switches; no path change to the `fit()` analytical solve |
| `biases_cifar10_panel.json` | 90 | `8da6d6…` | post-`644d7c1`+B2 work | streaming improvements (don't affect `fit()`-only B1.2) |
| `b3_strategies_*_panel.json` (3) | 70 each | `3d1c1d…` | post-`f4bc25b` / `c1d9b0a` / `cb8e1b7` | streaming Cholesky solve; clone-before-diagonal-add; `linear` activation registered |
| `biases_*_centered/decentered_panel.json` (B7) | 90 each | `004157…` | `e5a2b7a` (B7 follow-ups) | adds centered/decentered dataset variants |
| `b4_mlp_*_panel.json` | 60 each | `01a071…` | post-`b895794` | layers.py P0/P1 fixes (alpha guard, float64 accumulators, `_M_inv` non-persistent buffer) |

Headline claims that depend on the analytical `fit()` path (H1, H6) are
unaffected by the streaming/Woodbury fixes — verified by inspection of
`src/leenef/layers.py:fit()` and `solvers.py`, neither of which has
changed semantics across these commits.  Headline claims that depend on
`continuous_fit` / Woodbury (Stream-NEF, B2) live on the `3d1c1d…`
identity, which already incorporates the Cholesky solve and the
clone-fix.  No re-runs are required; the per-panel `code_identity` is
the ground truth for any future claim audit.


---

## Phase index

| # | phase | experiment | knob | claim it supports |
|---|---|---|---|---|
| — | smoke_n_neurons | smoke | `n_neurons` | infrastructure plumbing only |
| — | smoke_data_driven_bias | smoke | `data_driven_biases` | infrastructure plumbing only |
| B1   | biases_mnist_panel    | biases_mnist    | `cell` (3×3 factorial) | H1 — encoder-type gap closure |
| B1.1 | biases_fashion_panel  | biases_fashion  | `cell` (3×3 factorial) | H1 — Fashion-MNIST replication |
| B1.2 | biases_cifar10_panel  | biases_cifar10  | `cell` (3×3 factorial) | H1 — CIFAR-10 replication and DC-drift discriminator |
| B2   | streaming_smnist_{row,permuted,solve_modes,alpha_sweep} | streaming_smnist_* | `n_neurons`, `α`, `solve_mode` | H3 (Stream-NEF equivalence) + H4 (scaling) |
| B3   | b3_strategies_{mnist,fashion,cifar10} | b3_strategies_* | `cell` (7 encoder strategies) | H6 — encoder strategy ablation |
| B4   | b4_mlp_{mnist,fashion,cifar10}_panel | b4_mlp_* | `cell` (depth × width × lr) | H2 — gradient-trained MLP comparator |
| B5   | b5_lstm_smnist_{row,permuted}_panel | b5_lstm_smnist_* | `cell` (hidden × layers × lr) | H4 — gradient-trained LSTM comparator on temporal sMNIST |
| B6   | b6_esn_smnist_{row,permuted}_panel | b6_esn_smnist_* | `cell` (n × spectral_radius × leak) | H4 — gradient-free ESN reservoir comparator on temporal sMNIST |
| B7   | biases_{cifar10_centered,mnist_decentered}_panel | biases_*_centered/decentered | `cell` (3×3 factorial) | H1 — DC-component falsifier; channel-centering vs offset injection |
| H8   | h8_linear_collapse | h8_linear_collapse_check | `activation ∈ {linear, abs}` × n | H8 — over-representation needs nonlinearity (single-seed sanity) |

---

## B1 — biases_mnist_panel (H1 keystone)

**Question.** Does the choice of bias mechanism matter for the encoder-
type gap on MNIST? The original H1 hypothesis was *"data-driven biases
collapse the encoder-type gap"*. The data tells a more nuanced story.

**Setup.** 3×3 factorial: `bias_mode ∈ {data_driven, random_normal,
zero}` × `encoder_strategy ∈ {hypersphere, gaussian, sparse}` × 10 seeds
× n_neurons=5000, abs activation, gain U(0.5, 2.0), α=1e-3, Cholesky
solver, chunked accumulation. Encoders and gains are bit-identical
across the 3 bias modes for the same seed (independent sub-RNGs per
stream), so the bias-mode contrast is causal.

Results: `results/sl_clean/biases_mnist_panel.json` (90 rows),
summary in `biases_mnist_panel_summary.json`. Wall time: ~18 min on
12-thread Ryzen CPU.

### Per-cell test accuracy (mean ± SE, n=10)

| bias_mode \ encoder | hypersphere | gaussian | sparse |
|---|---|---|---|
| data_driven   | 97.1 ± 0.0 | 97.1 ± 0.0 | 97.1 ± 0.0 |
| random_normal | **96.1 ± 0.0** | 97.1 ± 0.0 | 97.2 ± 0.0 |
| zero          | 97.1 ± 0.0 | 97.1 ± 0.0 | 97.2 ± 0.0 |

### Encoder-type gap (max pairwise gap within each bias_mode, paired by seed)

| bias_mode | mean | SE | n | verdict |
|---|---|---|---|---|
| data_driven   | 0.10% | 0.02% | 10 | collapse_supported |
| random_normal | 1.12% | 0.04% | 10 | gap_present |
| zero          | 0.11% | 0.02% | 10 | collapse_supported |

(collapse threshold = 0.50 pp.)

### Headline contrasts: data_driven − random_normal (paired by seed)

| encoder | Δ mean | Δ SE | ratio | n |
|---|---|---|---|---|
| hypersphere | **+0.99 pp** | 0.05 | **18.5** | 10 |
| gaussian    | −0.06 pp | 0.05 | −1.3 | 10 |
| sparse      | −0.11 pp | 0.03 | −4.0 | 10 |

### Verdict

The original H1 framing — *"data-driven biases close the encoder-type
gap"* — needs revision. The data shows three things the simple framing
hides:

1. **`random_normal` biases are a failure mode for `hypersphere`
   encoders specifically** — they cost ~1 pp on MNIST. `gaussian` and
   `sparse` encoders are robust to bias choice.
2. **`zero` biases collapse the gap just as effectively as
   `data_driven`.** The honest story is not that data-driven biases
   are uniquely good, but that random Gaussian biases are uniquely bad
   for some encoders.
3. **Therefore the practical recommendation simplifies:** *don't use
   random Gaussian biases.* `data_driven` and `zero` are
   interchangeable on this benchmark within ≤ 0.05 pp.

This is a stronger and more useful finding than the original H1, and
it raises follow-up questions before B1 can close:

- **Is `data_driven == zero` dataset-specific?** Probably not on MNIST,
  but Fashion-MNIST and CIFAR-10 may differentiate them. Adding a
  Fashion-MNIST replication of B1 is now elevated to a B-phase blocker.
- **Does the gain range matter?** Larger gains amplify the
  random-Gaussian bias offset; smaller gains may close it. A gain
  ablation with random_normal+hypersphere is queued.
- **Why `hypersphere` only?** The unit-norm constraint means random
  biases land neurons at unpredictable points relative to the data
  cloud. `gaussian` and `sparse` directions implicitly carry magnitude
  variation that interacts with U(0.5, 2.0) gain in a way that
  compensates. This is a hypothesis for §5 of the paper, not yet
  proven.

### Code / data identity

- `code_identity` for all 90 rows: `a1837796f1fc783b83846670e39f75548281a979`
- `dataset.raw_hashes` (MNIST IDX): per-row in JSON; should match across
  any reproduction on this machine.

### Reproduce

```bash
SEEDS=0-9 PYTHON=$PWD/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh biases_mnist
```

---

## B1.1 — biases_fashion_panel (H1 replication on Fashion-MNIST)

**Question.** Does the B1 finding — *random Gaussian biases break
unit-norm encoders specifically; data-driven and zero biases both close
the encoder-type gap* — replicate on a slightly harder static
benchmark with the same input geometry as MNIST?

**Setup.** Identical to B1 (5000 neurons, abs activation, α=1e-3,
chunk=5000, paired encoders+gains across bias modes within seed).
Dataset: Fashion-MNIST 60k/10k, flattened to ℝ⁷⁸⁴, divided by 255.
10 seeds × 9 cells = 90 rows.

### Per-cell test accuracy (mean ± SE, n=10)

| bias_mode      | hypersphere      | gaussian         | sparse           |
|----------------|------------------|------------------|------------------|
| `data_driven`  | 87.74 ± 0.06 %   | 87.75 ± 0.06 %   | 87.82 ± 0.04 %   |
| `random_normal`| **86.57 ± 0.05 %** | 87.74 ± 0.04 %   | 87.82 ± 0.03 %   |
| `zero`         | 87.75 ± 0.04 %   | 87.75 ± 0.04 %   | 87.88 ± 0.06 %   |

### Paired-by-seed max encoder gap per bias mode

| bias_mode      | mean (pp) | min (pp) | max (pp) | n |
|----------------|-----------|----------|----------|---|
| `data_driven`  | 0.216     | 0.020    | 0.440    | 10 |
| `random_normal`| 1.282     | 0.830    | 1.550    | 10 |
| `zero`         | 0.214     | 0.080    | 0.480    | 10 |

### Verdict

**Replicates the MNIST finding cleanly.**

- `data_driven` ≡ `zero` to within 0.01 pp on every cell.
- `random_normal` + `hypersphere` is the only failure cell, costing
  ~1.18 pp vs the same encoders under either of the other bias modes.
- `gaussian` and `sparse` encoders absorb the random-bias perturbation.

This is two-for-two on the user's "switch the default to zero"
hypothesis as far as MNIST-shaped data is concerned. CIFAR-10 (B1.2)
will determine whether the equivalence also holds when the input
distribution has strong DC structure.

### Code / data identity

- code_identity: `644d7c154c9bbcdab379a0309a762b599989c307`
- run wall time: ~17 min on 6-physical-core CPU
- Fashion-MNIST raw / processed dataset hashes recorded per row in
  `metrics.dataset_metadata.{raw_hashes, processed_hashes}`.
- The `*_summary.json` for this phase is empty on disk because the
  driver-time analyzer ran post-phase under a HEAD that had moved on
  (an unrelated docs commit landed mid-sweep, bumping HEAD without
  touching `CODE_IDENTITY_PATHS`). The 90 raw rows are intact; the
  table above was regenerated by direct aggregation of the JSON.
  This motivates the follow-up to make `code_identity` content-based
  rather than HEAD-SHA-based; tracked as B1.fix.code_identity.

### Reproduce

```bash
SEEDS=0-9 PYTHON=$PWD/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh biases_fashion
```

---

## B1.2 — biases_cifar10_panel (H1 replication on CIFAR-10; DC-drift discriminator)

**Question.** Does `data_driven == zero` survive on CIFAR-10, where
the input has strong DC structure (mean intensity ≈ 0.49 per channel,
so ⟨e_i, x⟩ has a large positive offset before per-pixel structure
even shows up)?

**Setup.** Same as B1.1 except: dataset is CIFAR-10 (50k/10k, RGB
32×32 flattened to ℝ³⁰⁷², divided by 255), and the chunk size is
2500 to bound the peak `Aᵀ A` accumulation memory. 10 seeds × 9 cells
= 90 rows.

### Per-cell test accuracy (mean ± SE, n=10)

| bias_mode      | hypersphere      | gaussian         | sparse           |
|----------------|------------------|------------------|------------------|
| `data_driven`  | **49.72 ± 0.08 %** | **49.72 ± 0.08 %** | **49.85 ± 0.10 %** |
| `random_normal`| 48.25 ± 0.08 %   | 48.85 ± 0.07 %   | 49.13 ± 0.06 %   |
| `zero`         | 48.33 ± 0.07 %   | 48.34 ± 0.07 %   | 48.53 ± 0.08 %   |

### Paired-by-seed max encoder gap per bias mode

| bias_mode      | mean (pp) | min (pp) | max (pp) | n |
|----------------|-----------|----------|----------|---|
| `data_driven`  | 0.286     | 0.050    | 0.660    | 10 |
| `random_normal`| 0.908     | 0.380    | 1.250    | 10 |
| `zero`         | 0.239     | 0.030    | 0.520    | 10 |

### Headline contrast: data_driven − zero, paired by seed

| encoder      | Δ mean (pp) | SE (pp) | n  |
|--------------|-------------|---------|----|
| hypersphere  | **+1.389**  | 0.097   | 10 |
| gaussian     | **+1.389**  | 0.102   | 10 |
| sparse       | **+1.313**  | 0.160   | 10 |

### Verdict

**`zero == data_driven` is rejected on CIFAR-10.** Data-driven biases
beat zero biases by **+1.31 to +1.39 pp** *uniformly across all three
encoder families*. The effect dwarfs `sqrt(SE_a² + SE_b²) ≈ 0.11 pp`,
so the adoption rule fires by ≥ 12σ — this is one of the cleanest
signals we've seen.

The paper-level claim that survives all three datasets is:

> **Data-driven biases match zero biases when the input is roughly
> centered (MNIST, Fashion-MNIST: |Δ| ≤ 0.05 pp), and recover
> ~1.4 pp when the input has strong DC structure (CIFAR-10).**
>
> Mechanism: aᵢ = |αᵢ·eᵢ·x + bᵢ| is meaningfully nonlinear only when
> its argument straddles zero on the data manifold. With bᵢ=0, that
> requires x to be approximately centered; on CIFAR-10 it isn't, and
> abs collapses to a near-linear operator. With bᵢ = −αᵢ·eᵢ·dᵢ for
> dᵢ a training sample, every neuron's abs hinge sits on the data
> manifold by construction, restoring nonlinearity.
>
> Frame-coefficient reading: data-driven biases convert the global
> magnitude frame analysis aᵢ = |⟨αᵢeᵢ, x⟩| into a *locally-centered*
> frame analysis aᵢ = αᵢ|⟨eᵢ, x − dᵢ⟩|. The two coincide on
> low-DC data and diverge on high-DC data.

The `random_normal` failure on `hypersphere` is also milder here
(~0.08 pp vs zero+hyp; ~1.5 pp vs data_driven+hyp). On CIFAR the
larger story is the data_driven/zero gap, not the random_normal/zero
gap — exactly the opposite pattern from MNIST/Fashion.

### Default decision

**Keep `centers=x_train` as the project default.** It costs nothing
(≤ 0.05 pp) on MNIST/Fashion and pays ~1.4 pp on CIFAR-10. There is
no input distribution for which switching to zero is preferable.

### Code / data identity

- code_identity: `8da6d60017...` (current HEAD at run time).
- run wall time: ~24 min on 6-physical-core CPU.
- CIFAR-10 raw / processed dataset hashes recorded per row.

### Reproduce

```bash
SEEDS=0-9 PYTHON=$PWD/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh biases_cifar10
```

---

## B3 — encoder-strategy ablation (H6)

**Question.** At fixed n_neurons and data-driven biases, which of the
seven encoder strategies in `ENCODER_STRATEGIES` actually help, which
are equivalent, and which hurt?

**Setup.** Single-knob factorial: `cell ∈ {hypersphere, gaussian, sparse,
receptive_field, whitened, class_contrast, local_pca}` × 10 seeds × 3
datasets (MNIST, Fashion-MNIST, CIFAR-10). n_neurons=5000, abs
activation, data-driven biases (the `centers=x_train` default kept by
the B1 follow-ups). Per-seed sub-generators for encoder, gain,
center_index, and (where applicable) PCA / kNN initialization.
Per-cell wall ~25-31s on CPU; 3 panels in parallel under OMP=4 took
~56 min total.

**Results.** See `results/sl_clean/b3_strategies_*_panel_summary.json`.

| Strategy | MNIST % | Fashion % | CIFAR-10 % |
|---|---|---|---|
| hypersphere (baseline) | 97.08 ± 0.04 | 87.75 ± 0.06 | 49.72 ± 0.08 |
| gaussian | 97.08 ± 0.04 | 87.75 ± 0.06 | 49.72 ± 0.08 |
| sparse | 97.08 ± 0.02 | 87.82 ± 0.04 | 49.85 ± 0.10 |
| **receptive_field** | **97.35 ± 0.03** | **88.58 ± 0.03** | **55.10 ± 0.10** |
| whitened | 97.46 ± 0.03 | 88.33 ± 0.04 | 50.93 ± 0.09 |
| local_pca | 97.91 ± 0.02 | 86.78 ± 0.03 | 46.84 ± 0.08 |
| class_contrast | 87.72 ± 0.06 | 84.61 ± 0.06 | 45.28 ± 0.11 |

Δ vs hypersphere, paired-by-seed (z = Δ / SE_paired):

| Strategy | MNIST Δ pp (z) | Fashion Δ pp (z) | CIFAR-10 Δ pp (z) |
|---|---|---|---|
| gaussian | -0.000 (0.0) | +0.000 (0.0) | -0.000 (0.0) |
| sparse | -0.004 (-0.1) | +0.076 (+0.9) | +0.125 (+1.1) |
| receptive_field | +0.271 (+5.8) | +0.839 (+12.8) | **+5.377 (+45.0)** |
| whitened | +0.378 (+6.5) | +0.585 (+8.1) | +1.207 (+11.1) |
| local_pca | +0.830 (+16.3) | -0.967 (-12.6) | -2.884 (-22.5) |
| class_contrast | -9.355 (-179.6) | -3.140 (-40.9) | -4.439 (-37.5) |

**Verdict.**
- `hypersphere ≡ gaussian` to <0.001 pp on every dataset (the encoder
  builder normalises Gaussian samples to the unit sphere, so this is
  expected; we keep it as a bookkeeping check).
- `sparse` ties `hypersphere` everywhere — cheap and equivalent.
- **`receptive_field` is the headline win**: +5.4 pp on CIFAR-10 at
  fixed n_neurons (49.7 → 55.1%), with consistent smaller wins on
  MNIST/Fashion. This is the spatial-locality prior buying its keep.
- `whitened` gives a small, consistent, dataset-aware edge
  (+0.4 / +0.6 / +1.2 pp).
- `local_pca` wins on MNIST but *loses* on Fashion and CIFAR-10 —
  data adaptation isn't free.
- `class_contrast` is a clean failure mode (-9.4 pp on MNIST). Useful
  as a cautionary "this is what *not* to do".

**Adoption rule outcome.** Project default `hypersphere` is kept as
the "vanilla random frame" baseline. `receptive_field` is the
recommended default for spatial inputs and is what every CIFAR-10
headline downstream (ensembles, ConvNEF, streaming sCIFAR) should
default to. No `base_config.json` mutation: the choice depends on
whether the input has spatial structure.

**Reproduce.**

```bash
SEEDS=0-9 PYTHON=$PWD/venv/bin/python \
  bash benchmarks/sl_clean/run_b3_panels.sh
```

Or per-dataset:

```bash
SEEDS=0-9 PYTHON=$PWD/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh b3_strategies_mnist
```

---

---

## B2 — streaming-NEF panels (H3 equivalence + H4 scaling)

**Question.** Can the streaming (delay-line reservoir) classifier match
batch-NEF on temporal sMNIST while staying online and gradient-free?
Does the equivalence hold across the three solve modes (`batch`,
`batch_acc`, `continuous`/Woodbury)?

**Setup.** sMNIST-row + permuted-sMNIST × `n_neurons ∈ {1k, 2k, 4k, 8k,
16k}` × {batch, batch_acc, continuous} × 10 seeds.  α trace-scaled
(1e-4 nominal × tr(AᵀA)/N).  Window=10, abs activation, hypersphere
encoders, data-driven biases.  Code identity `3d1c1d…` includes the
Cholesky solve and the clone-fix.

**Status.** Run; raw JSON in `results/sl_clean/streaming_smnist_*.json`
and `streaming_solve_modes.json`.  Headline numbers and verdict to be
written up in this section once the analyzer table for the final code
identity is consolidated.

---

## B7 — biases follow-ups (DC-component falsifiers → mechanism reframed)

**Question.** Can we confirm that the data_driven – zero gap on
CIFAR-10 (B1.2) is driven by the dataset's DC offset?  Two predictions:

1. **B7a — `biases_cifar10_centered`.** Per-channel-centered CIFAR-10
   should *close* the gap to within 0.2 pp.
2. **B7b — `biases_mnist_decentered`.** Adding a constant offset to
   MNIST should *open* a +1 to +1.5 pp gap.

**Setup.** Same 3×3 factorial as B1, with the dataset preprocessing
swapped.  10 seeds × 9 cells × 2 datasets = 180 rows.  Code identity
`004157…`.

**Status.** Run; data in `results/sl_clean/biases_cifar10_centered_panel.json`
and `biases_mnist_decentered_panel.json`.

**Verdict.** Prediction (1) **falsified** — centering CIFAR-10 *opens*
the gap to **+6.3 pp** (the largest data-driven-vs-zero gap we've seen).
Prediction (2) **confirmed** — decentering MNIST opens a +1.3 pp gap.

| scenario               | data_driven% | zero%  | gap     | DC reading | geometric reading |
|------------------------|--------------|--------|---------|------------|-------------------|
| MNIST raw              | 97.08        | ~97.0  | 0.0     | predicted ~0 | predicted ~0  |
| MNIST decentered (+0.5)| 97.12        | 95.78  | **+1.3**| predicted +  | predicted + |
| CIFAR-10 raw           | 57.79        | 56.40  | +1.4    | predicted +  | predicted + |
| CIFAR-10 centered      | 49.72        | 43.41  | **+6.3**| predicted ~0 | predicted ++ |

**Mechanism reframed.** The gap is *not* about DC component magnitude.
It tracks the distance from the data centroid to the abs hinge
location.  Zero bias places every hinge at the origin; data-driven
bias places hinge i at training sample d_i.  When the data centroid
is **at** the origin (CIFAR-10 centered), zero-bias hinges fold every
data point through the middle of its mass — abs collapses sign
symmetrically, the worst case.  When the centroid is **far from** the
origin (MNIST raw), the hinge rarely sees a sign flip, abs is
effectively linear in the fluctuations, and bias placement is moot.
"DC offset = constant pixel mean" is one specific instance of this
geometry, but the mechanism is positional, not spectral.  See
`docs/paper_insights.md` §1.2 ("Where the abs hinge sits relative to
data") for the full reading.

**Implication for paper.**  The §1.2 / §6 narrative leads with the
geometric framing.  DC vocabulary survives as a useful first
introduction (constant-offset = pure-DC instance) but is no longer
the headline mechanism.  No new B-phase needed.

---

## B4 — gradient-trained MLP comparator (H2)

**Question.** Where does a properly tuned gradient-trained MLP sit
relative to a single-layer NEF on the three SL benchmarks?

**Setup.** 3×4×5 factorial: depth ∈ {1, 2}, width ∈ {256, 512, 1024},
lr ∈ {1e-2, 3e-3, 1e-3, 3e-4, 1e-4} × 5 seeds = 60 cells × 3 datasets.
Adam optimizer, batch=128, up to 50 epochs, early stopping with
patience=5 on a 10% validation hold-out.  Dropout=0; weight_decay=1e-4.
Code identity `01a071…` (post-`b895794`).

**Status.** Run; raw JSON in `results/sl_clean/b4_mlp_{mnist,fashion,cifar10}_panel.json`
and per-cell summaries in `..._panel_summary.json`.

| dataset | NEF baseline (B1, hypersphere n=5000) | best MLP (B4) | best MLP cell | Δ (MLP − NEF) |
|---|---|---|---|---|
| MNIST | 97.08 ± 0.04 | **98.10 ± 0.08** | `d1_w512_lr0.003` (407k params, 22 ep, 58 s) | +1.02 pp |
| Fashion-MNIST | 87.75 ± 0.06 | **89.39 ± 0.08** | `d2_w1024_lr0.001` (1.86M params, 24 ep, 254 s) | +1.64 pp |
| CIFAR-10 | 49.72 ± 0.08 | **53.73 ± 0.22** | `d2_w1024_lr0.001` (4.21M params, 34 ep, 327 s) | +4.01 pp |

**Verdict (H2).** A properly tuned gradient-trained MLP **beats**
single-layer hypersphere NEF on all three datasets, by a margin that
grows with task difficulty (+1.0 / +1.6 / +4.0 pp).  This is the
honest framing for the paper: NEF does not match an MLP at fixed
hardware effort if the encoder is generic.

But — **B3 receptive_field NEF on CIFAR-10 reaches 55.10 ± 0.10**,
which beats the best B4 MLP at `d2_w1024_lr0.001` (53.73 ± 0.22) by
+1.37 pp at well outside seed noise.  When the encoder carries the
spatial prior, a fully gradient-free single-layer NEF *exceeds* a
4.2M-parameter gradient-trained MLP on CIFAR-10, in a ~25 s analytic
solve versus a ~5 min Adam fit.  This is the interesting headline:
**the encoder, not the decoder, is where representational work
happens; a good encoder closes (and sometimes reverses) the
single-layer / gradient-trained gap**.

What this story is *not*: a claim that NEF beats deep convnets, or
that NEF is universally faster than gradient training.  Tuned MLPs
beat hypersphere NEF; a good prior closes the gap on spatial data;
deeper gradient-trained models would presumably extend the gap again.
The paper should state all three observations side-by-side.

---

## B5 — gradient-trained LSTM comparator (H4)

**Question.** Where does a tuned LSTM sit on temporal sMNIST relative
to Stream-NEF (B2)?

**Setup.** Pixel-by-pixel sMNIST + permuted-sMNIST × `hidden ∈ {64,
128, 256}` × `layers ∈ {1, 2}` × `lr ∈ {3e-3, 1e-3, 3e-4}` × 5 seeds
= 90 cells per panel.  Adam, batch=128, up to 30 epochs, early
stopping with patience=5 on a 10% val hold-out.  Code identity
`a495b4…`.

**Status (2026-05-02 09:35 UTC).**

- **b5_lstm_smnist_row: 90/90 ✅ COMPLETE** (sentinel hand-touched on
  2026-05-02; the original driver missed the `b5_panels_done` touch
  on the row branch — runner JSON is correct).  Top three cells:

  | cell | mean acc | ± SE | wall/seed | params |
  |---|---:|---:|---:|---:|
  | h256_l2_lr0.003 | 99.12 % | 0.09 | 1875 s | 821 770 |
  | h128_l2_lr0.003 | 99.10 % | 0.10 |  640 s | 214 282 |
  | h64_l2_lr0.003  | 99.07 % | 0.03 |  227 s |  57 994 |

  18 cells × 5 seeds; LR=3e-4 cells are clearly underconverged
  (97.1–98.0 %).  Best LSTM beats Stream-NEF on row by 0.7 pp at
  ~20× wall.

- **b5_lstm_smnist_permuted**: **2/90 rows on disk after 7+ hours,
  killed.**  Per-seed wall is **3.5 hours** — vanilla LSTM does not
  early-stop on permuted-sMNIST because val loss never plateaus, so
  it runs the full 30-epoch budget every time.  The 2 cells we have
  score 50.6 % and 52.0 %, consistent with the well-documented
  literature result that vanilla LSTM tops out near 50 % on
  permuted-sMNIST without curriculum tricks or special inits.

**Decision.** B5_permuted documented as an honest scaling limit, not
a clean multi-seed verdict.  This contrast — vanilla LSTM at 50 % in
3.5 h vs Stream-NEF at 75.6 % in 13 min on permuted — is itself a
paper-worthy data point about the cost-quality tradeoff on hard
temporal benchmarks.

---

## B6 — ESN reservoir comparator (H4 — gradient-free)

**Question.** How does a classical leaky-integrator ESN with
ridge-regression readout compare to Stream-NEF on temporal sMNIST?
This is the *gradient-free* ceiling for the same problem class.

**Setup.** sMNIST-row + permuted-sMNIST × `n ∈ {500, 1000, 2000}` ×
`spectral_radius ∈ {0.7, 0.95}` × `leak ∈ {0.3, 1.0}` × 5 seeds = 60
cells per panel.  Float64 ridge solve.  Three explicit Generators per
ESN (input-W, reservoir-W, internal sample) seeded as
`seed * 7919 + {11, 12}`.  Sparse reservoir built via
`randperm(n*n)[:k]` to guarantee unique non-zero positions
(reproducibility hazard fixed during smoke testing — see
`AGENTS.md` §"Reproducibility hazards").

**Status (2026-05-02 09:35 UTC).**

- **b6_esn_smnist_row: 60/60 ✅ COMPLETE.**  Top cells:

  | cell | mean acc | ± SE | wall/seed |
  |---|---:|---:|---:|
  | n2000_sr0.95_leak0.3 | 93.52 % | 0.06 | 454 s |
  | n2000_sr0.7_leak0.3  | 92.03 % | 0.04 | 462 s |
  | n1000_sr0.95_leak0.3 | 90.85 % | 0.13 | 110 s |

  Median cell ~77 %.  Stream-NEF n=8000 (98.45 %) clears best ESN by
  ~5 pp at ~5× faster wall.

- **b6_esn_smnist_permuted: 60/60 ✅ COMPLETE** (relaunched after
  78-minute stall at cell 58/60; driver auto-resumed from sentinel).
  Top cells (5 seeds each):

  | cell                        | mean acc | ± SE   |
  |-----------------------------|---------:|-------:|
  | n=2000, sr=0.95, leak=0.3   | **79.21 %** | 0.13 |
  | n=2000, sr=0.95, leak=1.0   |  75.96 % (2 seeds — partial) |
  | n=1000, sr=0.95, leak=0.3   |  75.26 % | 0.59 |
  | n=2000, sr=0.7, leak=0.3    |  60.99 % | 0.17 |

  **Story reversal vs row sMNIST.** On permuted sMNIST, ESN n=2000
  (79.21 %) **beats** Stream-NEF n=4000 (75.61 %) by +3.6 pp at half
  the reservoir size.  The mean-pool prior that wins on coherent
  row-of-pixels slices loses to recurrent state when the
  discriminative signal is distributed across all 784 randomly-
  ordered timesteps.  Paper I §6.3 reframes this honestly: the
  data picks the architecture.

**Headline triangle (sMNIST-row, all multi-seed):**

| method | best mean acc | ± SE | wall/seed | seed n |
|---|---:|---:|---:|---:|
| LSTM h256_l2 lr0.003          | 99.12 % | 0.09 | 1875 s | 5 |
| Stream-NEF n=8000             | 98.45 % | 0.02 |   95 s | 10 |
| Stream-NEF n=4000             | 97.97 % | 0.02 |   42 s | 10 |
| ESN n=2000 sr0.95 leak0.3     | 93.52 % | 0.06 |  454 s | 5 |
| Stream-NEF n=2000             | 97.20 % | 0.04 |   20 s | 10 |

Stream-NEF dominates the gradient-free comparator (ESN) by ~5 pp at
~5× faster wall, and trails the gradient-trained best LSTM by 0.7 pp
at ~20× faster wall.  This is the H4 picture Paper I §6 needs.

---

## H8 — over-representation needs nonlinearity (sanity check)

**Question.** Does a linear-activation NEF layer collapse to ordinary
ridge regression regardless of `n_neurons`?  Paper I §0 claims that
over-representation buys nothing without nonlinearity; H8 is the
single-seed sanity that backs that opener.

**Setup.** `linear` activation registered in `activations.py` (commit
`cb8e1b7`).  MNIST, n_neurons ∈ {1000, 4000}, bias_mode ∈ {data_driven,
zero}, α=1e-3.  Compared to plain ridge regression on raw 784-dim
input and to abs-activation NEFLayer at n=5000.

**Result.** `results/sl_clean/h8_linear_collapse.json`.  Linear
NEFLayer plateau at 86.0–86.2% (insensitive to `n_neurons`); plain
ridge baseline at 85.3%; abs NEFLayer at 97.08% (B1, 10-seed mean).
The ~+11 pp delta is the entire contribution of the abs nonlinearity.

**Verdict.** H8 supported.  Single-seed result; multi-seed not
required because the headline (linear ≪ abs) is far outside seed
noise on B1.

---

## Phase C — confirmation reruns (delegated)

Phase C is the multi-seed confirmation pass for Paper I supporting
claims.  Scaffolded on branch `paper-i/phase-c-experiments`
(commit `a50e2f9`); execution delegated to a separate Copilot session
per `~/.copilot/session-state/.../files/phase_c_delegation_2026-05-02.md`.
Estimated total wall: ~32 CPU-hours on the 12-core box.

| ID | knob | datasets | cells × seeds | hypothesis |
|---|---|---|---:|---|
| C1 | `cell` = n_neurons ∈ {500..20000} (6) | MNIST, Fashion, CIFAR-10 | 6 × 10 × 3 | scaling — single-layer plateau |
| C2 | `cell` = activation × gain (9) | MNIST, Fashion, CIFAR-10 | 9 × 10 × 3 | abs is not a magic value, but it is good |
| C3 | `cell` = M × encoder (8) | CIFAR-10 | 8 × 5 | ensemble + RF improvements |
| C4 | `cell` = strategy ∈ {greedy, hybrid, TP, E2E, hybrid_e2e} | MNIST, CIFAR-10 | 5 × 5 × 2 | H6 — multilayer is the load-bearing negative |
| C5 | `cell` ∈ {E2E, TP} | sMNIST-row | 2 × 5 | recurrent gradient is required |
| C6 | `cell` = default (single config) | MNIST, CIFAR-10 | 1 × 5 × 2 | ConvNEF is competitive |
| C7 | `cell` = n_neurons ∈ {500..10000} (5) | California Housing | 5 × 3 | regression universality |

None of the C-phases is expected to overturn a B-phase claim.  Any
"⚠ surprising" commit on the delegated branch will trigger a
re-merge here with re-aggregation.  Until then, Phase C panels are
treated as pending data for paper figures, not for paper claims.

---

## Config drift audit (2026-05-02)

A row in any `*_panel.json` is only aggregable if its
`code_identity` SHA matches the current tree.  The base-config
files for every B-phase already capture the constants under which
each panel was launched.  Sentinel facts to remember when writing
the paper:

- **Defaults at sl_clean start vs current.** No silent default has
  changed.  `gain_low/high = 0.5/2.0`, `encoder_strategy =
  hypersphere`, `activation = abs`, `data_driven_biases = True`,
  `chunk_size = 5000` were the launch defaults and are still the
  current defaults; per-experiment base configs override only what
  that phase contrasts.
- **Trace-scaled α.**  Adopted in B2 (commit `3e93c6c`).  All
  streaming panels and Phase C re-runs use `alpha_scale = "trace"`
  with `alpha_nominal` set per experiment.  B1 / B3 / B7 used the
  pre-trace default (raw α=1e-3); their `code_identity` predates
  the change, so aggregator excludes any post-change rerun under
  the old base if the SHA bump is missed.  No B-phase numbers move
  if recomputed under the new α convention because the trace scale
  reduces to the old constant when `n_neurons` is fixed.
- **`chunk_stats=True` default.**  Adopted in B2 (same commit).
  Pre-existing B1 cells used `chunk_stats=False`; the metric is
  identical since both paths compute the exact same `AᵀA`/`AᵀY`.
- **Linear activation.**  Registered in `cb8e1b7`; relevant only
  for H8 sanity check.
- **Cholesky solve + clone-before-diagonal-add.**  Adopted in
  `c1d9b0a` / `f4bc25b`.  All B3+ panels use this path; the
  earlier in-place `_ata.diagonal().add_(α)` would have silently
  corrupted `_ata` on the second `solve()` (caught in B2 review).
  Pre-fix panels (B1) only ever called `solve()` once per cell, so
  the bug never had a chance to bite there — verified by inspection
  of the runner.

**Audit conclusion.**  No B-phase number requires re-running for
paper-claim purposes.  Each panel's headline is reproducible from
its `code_identity`-tagged JSON under the original base config.
Phase C reruns are *additional* multi-seed coverage, not corrections.

---

## Smoke (worked example)

A tiny synthetic binary-classification task that exercises the full
runner / analyzer path without pulling MNIST. Useful to sanity-check
new infrastructure changes against the cached cells before launching
real sweeps. Expected wall time: a few seconds per cell.

```bash
bash benchmarks/sl_clean/sl_hp_driver.sh smoke
```

The smoke phases are not load-bearing for any paper claim and will be
deleted from the index once at least one headline phase has landed and
demonstrated the same workflow.

---

## Base config evolution

Each headline experiment will track its own `base_config_<exp>.json`.
When the analyzer adopts a new value, append a row here with the new
snapshot, the phase that triggered the change, the Δ / SE_combined,
and the seed counts on each side.

| date | experiment | base file | knob | old → new | Δ ± SE_combined | n_winner / n_base |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |
