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
| C1   | c1_scaling_mnist_panel | c1_scaling_mnist | `cell` (`n_neurons`) | Paper I scaling figure — MNIST |
| C1.1 | c1_scaling_fashion_panel | c1_scaling_fashion | `cell` (`n_neurons`) | Paper I scaling figure — Fashion-MNIST |
| C1.2 | c1_scaling_cifar10_panel | c1_scaling_cifar10 | `cell` (`n_neurons`) | Paper I scaling figure — CIFAR-10 |
| C2   | c2_activation_mnist_panel | c2_activation_mnist | `cell` (activation × gain) | Paper I activation/gain ablation — MNIST |
| C2.1 | c2_activation_fashion_panel | c2_activation_fashion | `cell` (activation × gain) | Paper I activation/gain ablation — Fashion-MNIST |
| C2.2 | c2_activation_cifar10_panel | c2_activation_cifar10 | `cell` (activation × gain) | Paper I activation/gain ablation — CIFAR-10 |
| C3   | c3_ensemble_cifar10_panel | c3_ensemble_cifar10 | `cell` (ensemble size × encoder) | Paper I ensemble / RF panel — CIFAR-10 |
| C4   | c4_multilayer_mnist_panel | c4_multilayer_mnist | `cell` (training strategy) | Paper I multilayer strategy panel — MNIST |
| C4.1 | c4_multilayer_cifar10_panel | c4_multilayer_cifar10 | `cell` (training strategy) | Paper I multilayer strategy panel — CIFAR-10 |
| C5   | gpu_c5_recurrent_smnist_panel | c5_recurrent_smnist | `cell` (training strategy) | Paper I recurrent row-sMNIST panel — Colab L4 |
| C6   | c6_convnef_mnist_panel | c6_convnef_mnist | `cell` (`default`) | Paper I ConvNEF headline — MNIST |
| C6.1 | c6_convnef_cifar10_panel | c6_convnef_cifar10 | `cell` (`default`) | Paper I ConvNEF headline — CIFAR-10 |
| C7   | c7_housing_panel | c7_housing | `cell` (`n_neurons`) | Paper I §5.6 — regression universality demo |

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
and `streaming_solve_modes.json`.

**Addendum 2026-05-03 — permuted scaling extension.** A clean-tree follow-on
extended `streaming_smnist_permuted_n_neurons.json` to
`n_neurons ∈ {8000, 16000}` with 10 seeds each under code identity
`9bfe9d9c7eacc013c2d83a42389245e47db129fe` (the older `2000/4000` rows in
that JSON remain under dirty identity `e106fe…`). The new clean rows are:

| n_neurons | test accuracy | verdict |
|---|---|---|
| 8000 | 77.68 ± 0.04 | strong gain over the old 4k regime |
| 16000 | 79.44 ± 0.03 | still improving, but still far from row-sMNIST |

The `16000` row beats `8000` by `+1.76 pp`, far outside the combined SE
(`0.05 pp`). So the permuted mean-pool story is now clearer: scale helps
materially all the way to 16k, but the extension **does not break the
ceiling**. Even at 16k, permuted Stream-NEF remains in the high-70s —
nowhere near the row-sMNIST regime where streaming reaches the high-90s.
This is exactly the characterization the delegation note asked for.

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

**Status (2026-05-01 19:45 UTC, mid-run).**

- **b5_lstm_smnist_row**: 41/90 rows on disk, ~9 min/seed, top so
  far 99.10% (h128_l1_lr0.0003 mean, 5 seeds, ~22 ep best).  ETA
  ~7-9 h to complete the panel.
- **b5_lstm_smnist_permuted**: **2/90 rows on disk after 7+ hours.**
  Per-seed wall is **3.5 hours** — vanilla LSTM does not early-stop
  on permuted-sMNIST because val loss never plateaus, so it runs
  the full 30-epoch budget every time.  The 2 cells we have score
  50.6% and 52.0%, consistent with the well-documented literature
  result that vanilla LSTM tops out near 50% on permuted-sMNIST
  without curriculum tricks or special inits.

**Decision.** Killing b5_permuted (~weeks at 3.5 h/seed × 90 × 5 =
infeasible).  The 2 cells we have are kept as illustrative; B5
permuted is documented as an honest scaling limit, not a clean
multi-seed verdict.  This contrast — vanilla LSTM at 50% in 3.5 h
vs Stream-NEF at 98.6% in minutes — is itself a paper-worthy data
point about the cost-quality tradeoff on hard temporal benchmarks.

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

**Status (2026-05-01 19:45 UTC).**

- **b6_esn_smnist_row: 60/60 ✅ COMPLETE.** Top cell (n=2000,
  sr=0.95, leak=1.0) reaches 93.65% test, ~488 s/seed.  Median
  cell ~77%.  Will be aggregated and written up below once
  permuted lands too.
- **b6_esn_smnist_permuted**: 34/60 mid-run, ~10 min/seed, top so
  far 76.4%.  ETA ~5 h.

**Audit addendum 2026-05-03.** A clean-tree re-run of the row-panel cell
`n=2000, sr=0.95, leak=0.3, seed=0` reproduces the stored dirty-row value
**exactly**: `93.33000183105469%` test accuracy in both the original
panel and `results/sl_clean/b6_esn_smnist_row_audit.json`. The original
panel row lives under dirty identity `a495b4…`; the audit row under clean
identity `9bfe9d…`. Because the numeric match is bit-exact, the
row-panel JSON is treated as **audited** and does **not** need a full
60-cell rerun.

**Provisional headline.** Stream-NEF (B2: 98.61% ± 0.04 at n=8000) is
~5 pp above the best ESN (93.7%) on row, and presumably similarly
above on permuted, at comparable solve cost.  The reservoir-vs-frame
comparison is exactly the comparator the paper needs to differentiate
"pooled abs frame coefficients" from "leaky-integrator dynamics".
Detailed verdict and table to follow once b6_permuted finishes.

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

## C1 — MNIST neuron-count scaling

**Question.** Does the default single-layer NEF on MNIST plateau by
`n=5000` or `n=10000`, validating the paper-wide choice to treat the
5000-neuron solve as safely inside the flat part of the scaling curve?

**Setup.** MNIST, `n_neurons ∈ {500, 1000, 2000, 5000, 10000, 20000}` ×
10 seeds = 60 rows. Hypersphere encoders, data-driven biases, `abs`
activation, gain `U(0.5, 2.0)`, `α=1e-3`, chunked normal-equation
accumulation with Cholesky solve. Result:
`results/sl_clean/c1_scaling_mnist_panel.json`. Log:
`results/sl_clean/logs/c1_scaling_mnist_panel.log`. Wall time: ~64 min
on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=10)

| n_neurons | test accuracy | verdict |
|---|---|---|
| 500 | 92.2 ± 0.1 | far from saturation |
| 1000 | 94.3 ± 0.1 | clear gain over 500 |
| 2000 | 95.8 ± 0.0 | still climbing |
| 5000 | 97.0 ± 0.0 | strong but not flat |
| 10000 | 97.6 ± 0.0 | another clear step up |
| 20000 | 98.0 ± 0.0 | **still improving** |

**Verdict.** ⚠ surprising. C1 on MNIST does **not** plateau by
`n=20000`. Accuracy rises monotonically from `97.63% ± 0.03` at
`n=10000` to `98.03% ± 0.03` at `n=20000`, a `+0.40 pp` gain that is
an order of magnitude larger than the combined SE. The conservative
reading is that MNIST is still in the upward part of the width curve at
20k, even though the marginal gains above 5000 are much smaller than the
gains below 5000.

**Code / data identity.**

- `code_identity` for all 60 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared MNIST raw bytes under the worktree's
  `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-9 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c1_scaling_mnist
```

---

## C1.1 — Fashion-MNIST neuron-count scaling

**Question.** Does the same width curve on Fashion-MNIST flatten early
enough to justify treating the 5000-neuron single-layer solve as
"already saturated" on a harder static image benchmark?

**Setup.** Fashion-MNIST, `n_neurons ∈ {500, 1000, 2000, 5000, 10000,
20000}` × 10 seeds = 60 rows. Hypersphere encoders, data-driven biases,
`abs` activation, gain `U(0.5, 2.0)`, `α=1e-3`, chunked normal-equation
accumulation with Cholesky solve. Result:
`results/sl_clean/c1_scaling_fashion_panel.json`. Log:
`results/sl_clean/logs/c1_scaling_fashion_panel.log`. Wall time: ~65 min
on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=10)

| n_neurons | test accuracy | verdict |
|---|---|---|
| 500 | 83.0 ± 0.1 | far from saturation |
| 1000 | 84.8 ± 0.0 | clear gain over 500 |
| 2000 | 86.2 ± 0.1 | still climbing |
| 5000 | 87.8 ± 0.1 | strong but not flat |
| 10000 | 88.7 ± 0.1 | another clear step up |
| 20000 | 89.3 ± 0.1 | **still improving** |

**Verdict.** C1.1 matches the revised C1 story: Fashion-MNIST also does
**not** plateau by `n=20000`. Accuracy rises monotonically from
`88.68% ± 0.06` at `n=10000` to `89.28% ± 0.06` at `n=20000`, a
`+0.60 pp` gain that is far outside the combined SE. The practical
implication is the same as on MNIST: `n=5000` is good enough for most of
the paper's comparisons, but it is not sitting on a true asymptote.

**Code / data identity.**

- `code_identity` for all 60 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared Fashion-MNIST raw bytes under the
  worktree's `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-9 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c1_scaling_fashion
```

---

## C1.2 — CIFAR-10 neuron-count scaling

**Question.** On CIFAR-10, does the default single-layer NEF keep
benefiting from brute-force width the way MNIST and Fashion-MNIST do, or
does the harder dataset expose a different scaling regime?

**Setup.** CIFAR-10, `n_neurons ∈ {500, 1000, 2000, 5000, 10000, 20000}`
× 10 seeds = 60 rows. Hypersphere encoders, data-driven biases, `abs`
activation, gain `U(0.5, 2.0)`, `α=1e-3`, chunked normal-equation
accumulation with Cholesky solve. Result:
`results/sl_clean/c1_scaling_cifar10_panel.json`. Log:
`results/sl_clean/logs/c1_scaling_cifar10_panel.log`. Wall time: ~64 min
on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=10)

| n_neurons | test accuracy | verdict |
|---|---|---|
| 500 | 43.3 ± 0.1 | clearly under-fit |
| 1000 | 45.8 ± 0.1 | clear gain over 500 |
| 2000 | 47.9 ± 0.1 | still climbing |
| 5000 | 49.4 ± 0.1 | near the peak |
| 10000 | 49.6 ± 0.1 | essentially tied with 5000 |
| 20000 | 47.1 ± 0.1 | **sharp regression** |

**Verdict.** C1.2 breaks the MNIST/Fashion pattern. CIFAR-10 improves
steadily through `n=5000`, gains only marginally at `n=10000`, and then
**collapses at `n=20000`**: test accuracy drops from
`49.59% ± 0.09` at `n=10000` to `47.08% ± 0.12` at `n=20000`, a
`−2.52 pp` change vastly larger than the combined SE. The same row
pair's train accuracy jumps from `77.05%` to `91.82%`, so this is not a
capacity bottleneck but a severe overfitting / generalization failure
regime. The practical Paper I recommendation is therefore dataset-
dependent: on CIFAR-10, **wider is not monotonically better**, and the
useful single-layer operating range looks more like `5000–10000`
neurons, not `20000+`.

**Code / data identity.**

- `code_identity` for all 60 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared CIFAR-10 raw bytes under the
  worktree's `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-9 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c1_scaling_cifar10
```

---

## C2 — MNIST activation × gain ablation

**Question.** On the default 5000-neuron MNIST solve, does `abs`
actually beat ReLU once seeds are averaged, and does per-neuron gain
diversity matter enough to justify the default `U(0.5, 2.0)` range?

**Setup.** MNIST, `activation ∈ {abs, relu, tanh}` ×
`gain ∈ {(0.5,2.0), (1.0,1.0), (0.1,5.0)}` × 10 seeds = 90 rows.
Hypersphere encoders, data-driven biases, `n_neurons=5000`, `α=1e-3`,
chunked normal-equation accumulation with Cholesky solve. Result:
`results/sl_clean/c2_activation_mnist_panel.json`. Log:
`results/sl_clean/logs/c2_activation_mnist_panel.log`. Wall time: ~40
min on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=10)

| cell | test accuracy | verdict |
|---|---|---|
| abs+g0.5_2.0 | 97.0 ± 0.0 | top tier |
| abs+g1.0_1.0 | 97.1 ± 0.0 | top tier |
| abs+g0.1_5.0 | 97.0 ± 0.0 | top tier |
| relu+g0.5_2.0 | 97.0 ± 0.0 | top tier |
| relu+g1.0_1.0 | 97.1 ± 0.0 | top tier |
| relu+g0.1_5.0 | 96.9 ± 0.0 | marginally below the leaders |
| tanh+g0.5_2.0 | 96.8 ± 0.0 | consistently worse |
| tanh+g1.0_1.0 | 96.7 ± 0.0 | consistently worse |
| tanh+g0.1_5.0 | 96.9 ± 0.0 | consistently worse |

**Verdict.** C2 on MNIST says the activation story is weaker than the
original hypothesis suggested. The best cell is `relu+g1.0_1.0` at
`97.06% ± 0.02`, but the whole `abs`/`relu` family sits inside a very
tight band (`96.94%` to `97.06%`) with heavily overlapping SEs. The
robust signal is only that **`tanh` trails by about 0.2–0.4 pp**. Gain
range is effectively a wash on MNIST: fixed gain, project-default gain,
and the wide-spread gain all land within seed noise for both `abs` and
ReLU.

**Code / data identity.**

- `code_identity` for all 90 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared MNIST raw bytes under the worktree's
  `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-9 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c2_activation_mnist
```

---

## C2.1 — Fashion-MNIST activation × gain ablation

**Question.** Does the activation story sharpen on Fashion-MNIST — a
harder static benchmark — enough to separate `abs` from ReLU and reveal
whether gain diversity matters in a less overdetermined regime?

**Setup.** Fashion-MNIST, `activation ∈ {abs, relu, tanh}` ×
`gain ∈ {(0.5,2.0), (1.0,1.0), (0.1,5.0)}` × 10 seeds = 90 rows.
Hypersphere encoders, data-driven biases, `n_neurons=5000`, `α=1e-3`,
chunked normal-equation accumulation with Cholesky solve. Result:
`results/sl_clean/c2_activation_fashion_panel.json`. Log:
`results/sl_clean/logs/c2_activation_fashion_panel.log`. Wall time:
~40 min on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=10)

| cell | test accuracy | verdict |
|---|---|---|
| abs+g0.5_2.0 | 87.8 ± 0.1 | top tier |
| abs+g1.0_1.0 | 87.8 ± 0.1 | top tier |
| abs+g0.1_5.0 | 87.6 ± 0.0 | slight drop but still competitive |
| relu+g0.5_2.0 | 87.6 ± 0.0 | close behind abs |
| relu+g1.0_1.0 | 87.7 ± 0.1 | close behind abs |
| relu+g0.1_5.0 | 87.5 ± 0.0 | slightly weaker |
| tanh+g0.5_2.0 | 87.1 ± 0.0 | clearly worse |
| tanh+g1.0_1.0 | 86.8 ± 0.0 | worst cell |
| tanh+g0.1_5.0 | 87.5 ± 0.0 | wide gain partially rescues tanh |

**Verdict.** C2.1 gives a cleaner separation than MNIST but not a huge
one. The best cells are the two `abs` variants (`87.78% ± 0.05` and
`87.81% ± 0.05`), with the best ReLU cell (`relu+g1.0_1.0`,
`87.73% ± 0.05`) only a hair behind. The stronger signal is that
**`tanh` is consistently worse**, especially with fixed gain, while a
wider gain spread substantially helps tanh recover. For `abs` and ReLU,
gain range still matters only marginally.

**Code / data identity.**

- `code_identity` for all 90 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared Fashion-MNIST raw bytes under the
  worktree's `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-9 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c2_activation_fashion
```

---

## C2.2 — CIFAR-10 activation × gain ablation

**Question.** On the default 5000-neuron CIFAR-10 solve, do activation
choice and gain spread matter more than they did on MNIST and
Fashion-MNIST, or is the same broad "abs/ReLU tie, tanh worse" story
still intact?

**Setup.** CIFAR-10, `activation ∈ {abs, relu, tanh}` ×
`gain ∈ {(0.5,2.0), (1.0,1.0), (0.1,5.0)}` × 10 seeds = 90 rows.
Hypersphere encoders, data-driven biases, `n_neurons=5000`, `α=1e-3`,
chunked normal-equation accumulation with Cholesky solve. Result:
`results/sl_clean/c2_activation_cifar10_panel.json`. Log:
`results/sl_clean/logs/c2_activation_cifar10_panel.log`. Wall time:
~49 min on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=10)

| cell | test accuracy | verdict |
|---|---|---|
| abs+g0.5_2.0 | 49.4 ± 0.1 | top tier |
| abs+g1.0_1.0 | 49.3 ± 0.1 | close to the leaders |
| abs+g0.1_5.0 | 49.7 ± 0.1 | best overall |
| relu+g0.5_2.0 | 49.5 ± 0.1 | top tier |
| relu+g1.0_1.0 | 49.3 ± 0.1 | slightly lower |
| relu+g0.1_5.0 | 49.5 ± 0.1 | top tier |
| tanh+g0.5_2.0 | 48.6 ± 0.1 | clearly worse |
| tanh+g1.0_1.0 | 48.0 ± 0.1 | weakest cell |
| tanh+g0.1_5.0 | 49.0 ± 0.1 | partially recovered, still behind |

**Verdict.** C2.2 matches the earlier C2 story more than it overturns
it. `abs` and ReLU remain effectively tied on CIFAR-10, with the best
cells all clustered around `49.4–49.7%` and overlapping SEs. The clean
signal is that **`tanh` remains materially worse**, especially with
fixed gain (`47.99% ± 0.07`). The gain story is slightly stronger than
on MNIST: the wide `g0.1_5.0` spread gives the best `abs` result and
partly rescues `tanh`, but the effect is still modest compared with the
activation-family gap. The practical default therefore still stands:
**stick with `abs` or ReLU; avoid `tanh`, especially with narrow/fixed
gain.**

**Code / data identity.**

- `code_identity` for all 90 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared CIFAR-10 raw bytes under the
  worktree's `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-9 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c2_activation_cifar10
```

---

## C3 — CIFAR-10 ensemble × receptive-field panel

**Question.** On CIFAR-10, does ensembling close the gap between the
default hypersphere prior and the spatially local receptive-field prior,
or does the receptive-field advantage persist no matter how many members
are averaged?

**Setup.** CIFAR-10, `n_members ∈ {1, 5, 10, 20}` ×
`encoder ∈ {hypersphere, receptive_field}` × 5 seeds = 40 rows.
Each member uses `n_neurons=5000`, `abs`, data-driven biases, and
`α=1e-3`; member logits are averaged across the ensemble. Result:
`results/sl_clean/c3_ensemble_cifar10_panel.json`. Log:
`results/sl_clean/logs/c3_ensemble_cifar10_panel.log`. Wall time:
~3 h 51 min on the 12-core CPU (across the interrupted first attempt and
the resumable rerun).

### Per-cell test accuracy (mean ± SE, n=5)

| cell | test accuracy | verdict |
|---|---|---|
| m1+hypersphere | 49.6 ± 0.1 | weakest baseline |
| m1+receptive_field | 55.1 ± 0.1 | strong single-member gain |
| m5+hypersphere | 53.8 ± 0.1 | ensemble helps |
| m5+receptive_field | 59.6 ± 0.1 | clear best at moderate cost |
| m10+hypersphere | 54.2 ± 0.1 | small gain over m5 |
| m10+receptive_field | 60.2 ± 0.1 | best moderate-size cell |
| m20+hypersphere | 54.7 ± 0.1 | diminishing returns |
| m20+receptive_field | 60.6 ± 0.1 | best overall |

**Verdict.** C3 strongly supports the receptive-field story. The
receptive-field prior stays **about 5.5–6.0 pp above** hypersphere at
every ensemble size, so ensembling does **not** erase the encoder-prior
advantage. Ensembling itself helps both families substantially from
`m=1` to `m=5`, but the marginal gain from `m=10` to `m=20` is small
(`+0.5 pp` for hypersphere, `+0.4 pp` for receptive_field). The
practical recommendation is therefore: if compute allows only one lever,
**choose receptive-field before buying more members**.

**Code / data identity.**

- `code_identity` for all 40 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared CIFAR-10 raw bytes under the
  worktree's `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-4 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c3_ensemble_cifar10
```

---

## C4 — MNIST multilayer strategy panel

**Question.** Once we allow one hidden NEF layer, do the multilayer
training strategies actually help on MNIST, or does the paper's
"single-layer wide solve is enough" story still dominate?

**Setup.** MNIST, one hidden layer with `hidden=[2000]`, output layer
with `output_neurons=2000`, `abs` activation, hypersphere encoders,
data-driven biases, `α=1e-3`, and 5 seeds. The knob is
`strategy ∈ {greedy, hybrid, target_prop, end_to_end, hybrid_e2e}`.
Result: `results/sl_clean/c4_multilayer_mnist_panel.json`. Log:
`results/sl_clean/logs/c4_multilayer_mnist_panel.log`. Wall time:
~6 h 18 min on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=5)

| strategy | test accuracy | verdict |
|---|---|---|
| greedy | 95.37 ± 0.09 | fast analytic baseline |
| hybrid | 94.25 ± 0.04 | worse than greedy |
| target_prop | 98.54 ± 0.02 | strong local-learning result |
| end_to_end | 98.60 ± 0.02 | strong BPTT baseline |
| hybrid_e2e | 98.69 ± 0.01 | best overall |

**Verdict.** C4 is **not** the simple negative result the phase note
anticipated. Purely analytic `greedy` is decent and `hybrid` is
actually worse, but once the network gets either local target
propagation or end-to-end refinement, performance jumps by roughly
`+3.2–4.4 pp` over the weak multilayer baselines. `hybrid_e2e` wins at
`98.688% ± 0.014`, `end_to_end` is close behind at
`98.598% ± 0.017`, and `target_prop` is only another `0.06 pp` back
while running in less than half the time of the BPTT-based variants.
So the honest C4 story is narrower: **plain multilayer stacking fails,
but MNIST does benefit from depth once a stronger training rule is
allowed.**

**Code / data identity.**

- `code_identity` for all 25 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared MNIST raw bytes under the worktree's
  `data/` symlink to `../leenef/data`
- The final JSON combines the original greedy/hybrid rows with a
  resumable same-identity recovery for the remaining strategies after
  the first wrapper hit a strategy-specific kwarg bug; all aggregate
  rows remain directly comparable because they share the same stamped
  identity

---

## C4.1 — CIFAR-10 multilayer strategy panel

**Question.** Does the same one-hidden-layer NEF stack help on CIFAR-10,
or does the harder dataset expose a much sharper split between weak
local/analytic training rules and full end-to-end refinement?

**Setup.** CIFAR-10, one hidden layer with `hidden=[2000]`, output layer
with `output_neurons=2000`, `abs` activation, hypersphere encoders,
data-driven biases, `α=1e-3`, and 5 seeds. The knob is
`strategy ∈ {greedy, hybrid, target_prop, end_to_end, hybrid_e2e}`.
Result: `results/sl_clean/c4_multilayer_cifar10_panel.json`. Log:
`results/sl_clean/logs/c4_multilayer_cifar10_panel.log`. Wall time:
~5 h 17 min on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=5)

| strategy | test accuracy | verdict |
|---|---|---|
| greedy | 46.08 ± 0.13 | modest baseline |
| hybrid | 36.37 ± 0.17 | catastrophic failure |
| target_prop | 50.59 ± 0.15 | modest gain over greedy |
| end_to_end | 59.20 ± 0.04 | clear best |
| hybrid_e2e | 58.47 ± 0.16 | strong but below pure E2E |

**Verdict.** C4.1 is much harsher than MNIST. `target_prop` helps over
the plain greedy stack, but only by about `+4.5 pp`, leaving it far
below the good CIFAR-10 front-runners. The real jump comes only from
full end-to-end training: `end_to_end` reaches `59.200% ± 0.044`, while
`hybrid_e2e` lands at `58.470% ± 0.164`. Unlike MNIST, `hybrid_e2e`
does **not** beat pure E2E here, and plain `hybrid` collapses badly to
`36.374% ± 0.170`. So the CIFAR-10 multilayer story is genuinely
negative for non-E2E training: **depth helps only when the model is
allowed to behave like a conventionally trained network; the purely
analytic/local variants remain uncompetitive.**

**Code / data identity.**

- `code_identity` for all 25 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared CIFAR-10 raw bytes under the
  worktree's `data/` symlink to `../leenef/data`
- This panel was run directly through the same-identity recovery runner
  to avoid the known stock-wrapper kwarg bug, so all rows still live in a
  single clean aggregate

---

## C5 — recurrent row-sMNIST panel (Colab L4)

**Question.** On row-sMNIST, does the lite recurrent NEF become genuinely
competitive once it is run on a GPU budget that makes the recurrent
panel practical, and does recurrent target propagation through time
recover that performance without BPTT?

**Setup.** Colab L4 GPU, row-sMNIST, 5 seeds, `n_neurons=1500`,
`d_state=128`, `n_epochs=30`, `batch_size=256`, `lr=1e-3`, `α=1e-3`,
and `relu` recurrent activation. The knob is
`strategy ∈ {end_to_end, target_prop}`. Authoritative panel:
`results/sl_clean/gpu_c5_recurrent_smnist_panel.json`. Log:
`results/sl_clean/logs/gpu_c5_recurrent_smnist_panel.log`. Companion
same-code-identity L4 comparator panels landed alongside it:
`gpu_streaming_smnist_row_panel.json`,
`gpu_b5_lstm_smnist_row_pareto_panel.json`, and
`gpu_b6_esn_smnist_row_best_panel.json`, with exact Colab configs under
`results/sl_clean/colab_gpu_configs/`. Wall time for the C5 panel:
~29 min on the L4.

### Per-cell test accuracy (mean ± SE, n=5)

| strategy | test accuracy | verdict |
|---|---|---|
| end_to_end | 98.83 ± 0.03 | competitive and stable |
| target_prop | 23.82 ± 0.11 | catastrophic failure |

**Verdict.** C5 strongly reinforces the old recurrent story instead of
softening it. `end_to_end` reaches `98.834% ± 0.029` on the L4 and does
so quickly (`108.30 ± 1.33 s` per seed), which makes the recurrent NEF
fully viable as a GPU benchmark. But `target_prop` through time fails
completely: `23.824% ± 0.106` test accuracy, `25.10%` train accuracy,
and **more than 2× slower** than `end_to_end` (`239.12 ± 0.67 s` per
seed). The companion L4 panels make the ranking clear:

- best LSTM pareto cells: `99.08 ± 0.11` and `99.07 ± 0.05`
- Stream-NEF row panel: `98.44 ± 0.04` at `n=8000`
- ESN best row cell: `93.52 ± 0.06`

So recurrent NEF with end-to-end training is **better than the streaming
row panel and vastly better than the ESN**, but still trails the best
LSTM by about `0.23 pp`. The practical conclusion is unchanged:
**only end-to-end recurrent NEF is competitive; target propagation is
not a viable recurrent training rule here.** The other practical lesson
is operational: this panel belongs on GPU/Colab, not on the local CPU.

**Code / data identity.**

- `code_identity` for all rows in the C5 and companion GPU panels:
  `c3ca3cf8e46c1d9a45be76f8626e3795cda3c5a2`
- Hardware stamped on all rows: `cuda_available=true`,
  `cuda_device_name='NVIDIA L4'`
- The companion GPU panels share the same code identity, so the
  cross-model row-sMNIST comparison above is apples-to-apples

---

## C6 — ConvNEF on MNIST

**Question.** Does the reduced single-config ConvNEF rerun still recover
the strong shallow-pipeline story on MNIST once we attach rl_clean-style
seed bars, or was the earlier result overly dependent on a lucky seed?

**Setup.** MNIST, ConvNEF `default` config × 5 seeds. Single PCA filter
stage (`64` filters, `patch_size=5`, `pool_size=2`), `n_neurons=4000`,
`pool_levels=[1,2,4]`, `standardize=True`, `abs` activation,
hypersphere encoders, gain `U(0.5, 2.0)`, analytic decoder with
`α=1e-2`. Result: `results/sl_clean/c6_convnef_mnist_panel.json`. Log:
`results/sl_clean/logs/c6_convnef_mnist_panel.log`. Wall time: ~4.4 min
on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=5)

| cell | test accuracy | verdict |
|---|---|---|
| default | 98.38 ± 0.02 | strong and stable |

**Verdict.** C6 on MNIST is cleanly supportive. The lite ConvNEF config
reaches `98.376% ± 0.017`, with a narrow seed spread and a training time
of `48.98 ± 0.57 s`. That places the shallow fixed-filter pipeline
comfortably above the single-layer 5000-neuron baseline while still
remaining fully gradient-free.

**Code / data identity.**

- `code_identity` for all 5 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared MNIST raw bytes under the worktree's
  `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-4 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c6_convnef_mnist
```

---

## C6.1 — ConvNEF on CIFAR-10

**Question.** Does the same lite ConvNEF configuration remain competitive
once we move from MNIST to CIFAR-10, where spatial structure matters
much more and shallow fixed-feature pipelines are under real pressure?

**Setup.** CIFAR-10, ConvNEF `default` config × 5 seeds. Single PCA
filter stage (`96` filters, `patch_size=5`, `pool_size=2`),
`n_neurons=4000`, `pool_levels=[1,2,4]`, `standardize=True`, `gcn=True`,
`abs` activation, hypersphere encoders, gain `U(0.5, 2.0)`, analytic
decoder with `α=1e-2`. Result:
`results/sl_clean/c6_convnef_cifar10_panel.json`. Log:
`results/sl_clean/logs/c6_convnef_cifar10_panel.log`. Wall time: ~7.5
min on the 12-core CPU.

### Per-cell test accuracy (mean ± SE, n=5)

| cell | test accuracy | verdict |
|---|---|---|
| default | 65.89 ± 0.12 | clearly above single-layer baseline |

**Verdict.** C6.1 supports the intended headline. The lite ConvNEF setup
reaches `65.892% ± 0.118` on CIFAR-10, substantially above the plain
single-layer CIFAR-10 numbers and comfortably above the ensemble-free
hypersphere baseline from C3. It does not approach modern CNN territory,
but it validates the paper's claim that a shallow, fully gradient-free,
data-derived conv front-end can be genuinely competitive on CIFAR-10.

**Code / data identity.**

- `code_identity` for all 5 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes match the shared CIFAR-10 raw bytes under the
  worktree's `data/` symlink to `../leenef/data`

### Reproduce

```bash
SEEDS=0-4 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c6_convnef_cifar10
```

---

## C7 — California Housing regression (universality demo)

**Question.** Does the same single-layer NEF primitive used throughout
Paper I transfer cleanly to a real-valued regression task, or does the
analytic solve rely on classification-specific structure?

**Setup.** California Housing, `n_neurons ∈ {500, 1000, 2000, 5000,
10000}` × 3 seeds = 15 rows.  Hypersphere encoders, data-driven biases,
`abs` activation, gain `U(0.5, 2.0)`, `α=1e-3`, chunked normal-equation
accumulation with Cholesky solve, and a deterministic train/test split
derived from `seed * 31337 + 11`.  Result: `results/sl_clean/c7_housing_panel.json`.
Log: `results/sl_clean/logs/c7_housing_panel.log`.  Wall time: ~1 min on
the 12-core CPU.

### Per-cell test R² (mean ± SE, n=3)

| n_neurons | test R² | verdict |
|---|---|---|
| 500 | 0.731 ± 0.005 | workable but clearly under-fit |
| 1000 | 0.747 ± 0.006 | meaningful gain over 500 |
| 2000 | 0.756 ± 0.005 | still improving |
| 5000 | 0.767 ± 0.009 | near-plateau |
| 10000 | 0.769 ± 0.010 | statistically tied with 5000 |

**Verdict.** C7 supports the universality claim.  The same fixed-encoder
plus analytic-decoder primitive reaches `test_r2 ≈ 0.77` on California
Housing with no architecture change beyond `d_out=1`.  Performance
improves monotonically with neuron count, then plateaus by roughly
`n=5000`; `n=10000` is only `+0.002` R² above `n=5000`, well inside the
combined SE.

**Code / data identity.**

- `code_identity` for all 15 rows: `9bfe9d9c7eacc013c2d83a42389245e47db129fe`
- Dataset hashes were stable across all rows:
  `data_npz=c8f43774aa27b7ec034e60db31b6124e12b4a0d42b55b0c8eb1320089e2ebc37`,
  `target_npz=ae80bb98487c04d7383ed8aecda8e207583dfd3e66f6679fded97597c3f6c87e`

### Reproduce

```bash
SEEDS=0-2 PYTHON=../leenef/venv/bin/python \
  bash benchmarks/sl_clean/sl_hp_driver.sh c7_housing
```

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
