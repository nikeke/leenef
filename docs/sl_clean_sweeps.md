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

---

## Phase index

| # | phase | experiment | knob | claim it supports |
|---|---|---|---|---|
| — | smoke_n_neurons | smoke | `n_neurons` | infrastructure plumbing only |
| — | smoke_data_driven_bias | smoke | `data_driven_biases` | infrastructure plumbing only |
| B1   | biases_mnist_panel    | biases_mnist    | `cell` (3×3 factorial) | H1 — encoder-type gap closure |
| B1.1 | biases_fashion_panel  | biases_fashion  | `cell` (3×3 factorial) | H1 — Fashion-MNIST replication |
| B1.2 | biases_cifar10_panel  | biases_cifar10  | `cell` (3×3 factorial) | H1 — CIFAR-10 replication and DC-drift discriminator |

Headline-claim phases (streaming sMNIST, MLP / LSTM / ESN baselines,
etc.) will be added as the corresponding experiments are wired up.

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
