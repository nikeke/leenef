# Paper I outline (Supervised Learning, ending with Stream-NEF)

This is the **structural skeleton** that the Paper I text will follow.
Numbers in `[…]` reference sections of `paper_insights.md` where the
content lives now.  The text-writing pass will turn each bullet into a
paragraph; we are not writing prose here.

The SSUP paper series' shared opener (the six principles, §0pre of insights)
goes in front of *all three* papers, so Paper I gets it as §1 and
Papers II–III repeat it verbatim.  This makes each paper readable in
isolation and signals continuity.

---

## §1 — The six principles of leenef  [insights §0pre]

Verbatim copy of §0pre.  No new content here; just the shared anchor.

## §2 — Why over-representation needs nonlinearity  [insights §0]

The opener for *this* paper (after the shared §1).  The argument:

- For any fixed encoder `E ∈ R^{n×d}`, `a = αEx` (linear) collapses
  through the linear decoder to `Wx`: `n` is irrelevant.
- abs/ReLU is the minimal repair; one piecewise-linear fold per
  neuron makes `aᵢ` an unsigned distance to the hyperplane
  `{eᵢ·z = 0}`.  Linear combinations of unsigned distances are
  piecewise-linear functions on `R^d`, which are universal
  approximators (cite Yarotsky for ReLU; abs reduces to ReLU via
  `|x| = ReLU(x) + ReLU(−x)`).
- ReLU and abs are equivalent for the analytic-solve setting: a
  ReLU neuron is half a fold; an abs neuron is two ReLU neurons
  with mirrored encoders, so the spans coincide once the decoder
  is solved.

**Frame-coefficient framing**:  set `αᵢeᵢ = uᵢ`.  Then
`aᵢ = |uᵢ · x|` is the magnitude of the `i`-th frame coefficient
of `x` in the (over-complete, possibly random) frame `{uᵢ}`.
The analytic decoder is the linear functional that reads off the
target from these magnitudes.  Section §3 of insights treats
this geometrically.

This section is not just metaphor — it predicts §6's empirical
finding (DC = mean image absorbed by linear decoder when biases
are zero, but not when activation is abs).

## §3 — Method: the NEF layer as one analytic step

Implementation-level.  Stays close to the code, no surprises.

- Forward: `a = G(α · (e · x − d) + b)` where `G` is the activation,
  `e` are unit encoders, `α` is a per-neuron gain, `d` is a
  data-driven center, `b = −α·(e·d)` is the corresponding bias.
- The default activation is `abs`; encoders are unit-vector
  hypersphere by default; gains drawn `U(0.5, 2.0)`; biases derived
  from training samples ("data-driven biases").
- Decoders solved by Tikhonov ridge: `D = (AᵀA + αI)⁻¹AᵀY`.  Three
  numerically-equivalent paths: `batch_acc` (one shot), `accumulate`
  (multi-batch normal-equation accumulator), `woodbury` (rank-k
  inverse update).  All three reach bit-identical results for
  reasonable `α`; H3 in §5.3 confirms.
- Pooling for time, additivity for tasks: forward-look only here;
  Papers II and III deliver.

## §4 — Regularization scales with the encoder span  [insights §2]

The most technical section.  Most-likely-misunderstood story.

- Naïve `α` (a fixed scalar) under-regularizes as `n_neurons` grows.
  Empirically: at n=8000 with α=10⁻⁶, accuracy is bimodal across
  seeds (some 98%, some 90%).
- The architectural fix is to scale α by `tr(AᵀA) / n` — the mean
  per-neuron self-product — which is a constant of the data + encoder
  but independent of `n` to first order.  This places
  `α_eff = α · tr(AᵀA)/n` on the same footing as the encoder span.
- Empirical α calibration sweep [results/sl_clean/streaming_alpha_sweep_summary.json]:
  - **5 alphas × 2 scaling modes × 2 n_neurons × 5 seeds = 100 cells.**
  - Trace-scaled α=10⁻⁴ wins both n=4000 (98.0%) and n=8000 (98.4%)
    with tight SE (≤0.04 pp).  Δ across n is +0.4 ± 0.05 pp.
  - Fixed α has a different optimum at each `n` (e.g. fixed α=10⁻²
    is best at n=4k but blows up to 87% at n=8k); confirms the
    architectural-α story.
  - Trace-scaled α=10⁻² (our previous default) over-regularises by
    ~2 OOM; explains why early streaming runs plateaued at 95–96%.

This section will own the **footgun-prevention claim**: any
practitioner copying our recipe needs trace-scaled α.

## §5 — Single-layer experiments: data-driven biases × encoders  [insights §1, §5]

The B1 panel.  Run rl_clean-style: 10 seeds, multi-config.

### §5.1 — Where the abs hinge sits relative to data  [insights §1.2, §6, §7 H1]

- Setup: NEFLayer with biases ∈ {data_driven, zero}.  With abs
  activation, the bias is the fold point of `aᵢ(x) = |α·eᵢ·x + bᵢ|`
  in input space — zero bias places the fold through the origin;
  data-driven bias `bᵢ = −α·eᵢ·dᵢ` places the i-th fold at training
  sample `dᵢ`.
- The natural prediction (pure-DC reading): for centered datasets
  (zero data centroid), zero biases place the fold at the data
  cloud → same as data-driven.  For off-centered datasets, fold
  rarely sees a sign flip → data-driven matters.
- B1 result on raw datasets matched the prediction *direction* —
  MNIST/Fashion gap ≤ 0.1 pp (off-centered), CIFAR-10 raw gap +1.4 pp
  (less off-centered).
- **B7 follow-ups falsified the spectral DC reading.**  Centering
  CIFAR-10 *opens* the gap to **+6.3 pp** — the largest data_driven
  vs zero gap we've seen, not the predicted ~0.  Decentering MNIST
  by +0.5 opens a +1.3 pp gap.  See `results/sl_clean/biases_*_panel.json`.
- **Reframed mechanism**: the gap depends on the geometric distance
  from the data centroid to the abs hinge (a hyperplane through
  origin for zero bias, through training sample for data_driven).
  Centroid AT origin = worst case (every hinge cuts through data
  mass symmetrically, abs collapses sign); centroid far from origin
  = best case for zero bias (hinge rarely fires, abs ≈ identity).
- The DC vocabulary survives as a useful *first lens*: a constant
  pixel mean is one specific instance of "data centroid not at
  origin".  But the headline mechanism is geometric/positional,
  not spectral.
- Implication for practice: zero biases work well **only if** the
  data has been pulled away from origin (typical raw image data
  with mean ≈ 0.1–0.5).  If the data is centered (a common ML
  preprocessing step), data-driven biases are essential.

### §5.2 — Encoder strategy is mostly orthogonal to bias choice  [insights §6]

- 7 strategies (hypersphere, gaussian, sparse, receptive_field,
  whitened, class_contrast, local_pca).
- B1 result: with data-driven biases, all 7 strategies cluster
  within ~0.5 pp on MNIST and Fashion.  Receptive_field wins by a
  hair on CIFAR-10 only because it embeds spatial locality, not
  because of analytic-solve behavior.
- Frame-prior interpretation: each strategy is a different prior
  over `{uᵢ}` directions.  Hypersphere = isotropic; whitened =
  data-covariance-aware; receptive_field = spatial-locality-aware.

### §5.3 — Solver-numerics equivalence (H3)  [insights §3]

- batch_acc, accumulate, woodbury produce equivalent decoders for
  reasonable α.
- Float64 in the streaming Woodbury inverse prevents drift; float32
  accumulate+solve is GPU-friendly and equally accurate.

## §6 — Streaming-NEF: time collapsed by pooling  [insights §1, §4, §5.7]

The headline empirical contribution.

### §6.1 — The architecture

- Delay-line (K-window unfold) + fixed encoder + abs + mean-pool +
  analytic decoder.  No recurrence.  No BPTT.
- Streaming variants: `batch_acc`, `accumulate`, `continuous_fit`
  (Woodbury rank-k).  All equivalent on test accuracy.

### §6.2 — Row sMNIST results

- 10 seeds × {n=2k, 4k, 8k} (B2 final panel).  Top: 98.61 ± 0.04
  at n=8k with trace-scaled α=10⁻⁴.
- Honest LSTM comparison (B5): tuned LSTM on row sMNIST tops
  ~99.1 ± seed-noise at h128_l1_lr=3e-4, ~9 min/seed (vs Stream-NEF
  ~30 s/seed at n=4k; CPU).  LSTM beats by ~0.5 pp at ~20× wallclock.
- Honest ESN comparison (B6): leaky-integrator ESN with ridge
  readout tops 93.65% on row at n=2000 sr=0.95 leak=1.0; fully
  gradient-free, ~488 s/seed.  Stream-NEF is **+5 pp above** the
  ESN ceiling at comparable analytic-solve cost — the relevant
  apples-to-apples comparison since both fit only a linear readout
  on top of fixed nonlinear features.

### §6.3 — Permuted sMNIST results

- 5 seeds × {n=2k, 4k} (B2 final panel).  Stream-NEF essentially
  unchanged from row (mean pooling is permutation-invariant).
- LSTM permuted (B5): vanilla LSTM does not early-stop on
  permuted-sMNIST (val loss never plateaus); single-seed runs take
  3.5 hours and converge to ~50% — consistent with the well-
  documented literature that vanilla LSTM fails this benchmark
  without curriculum/init tricks.  **Documented as honest scaling
  limit, not a clean multi-seed verdict** (we have 2 cells; the
  full 90-cell × 5-seed grid was infeasible at ~weeks of CPU).
- ESN permuted (B6): tops ~76% (still finishing).
- Stream-NEF (98.4–98.6%) sits 22+ pp above ESN, 48+ pp above
  vanilla LSTM at the practical wallclock budget.  Both contrasts
  go in the paper; the LSTM contrast is presented as
  cost-vs-quality at a fixed compute envelope, not as "we beat
  LSTM" (the literature has dozens of LSTM variants that solve
  permuted-sMNIST; see §7).

### §6.4 — Limits  [insights §4]

- Window K bounds the receptive field; periodicities longer than K
  are invisible to mean pooling.
- For non-stationary sequences (audio, video), mean-pool washes out
  per-position information.  This is the boundary where pooling
  stops carrying us and recurrence becomes necessary — the bridge
  to Paper III.

## §7 — Single-layer comparator: gradient-trained MLP (B4)

A separate section because it is the cleanest "where do we stand?"
benchmark for the single-layer NEF, *before* we get to streaming.

- B4 setup: 12 configs per dataset (depth ∈ {1,2}, width ∈
  {256,512,1024}, lr ∈ {1e-3,3e-3}) × 5 seeds.  Adam, batch=128,
  early stop with patience=5 on 10% val.
- Headline numbers (best MLP cell vs B1 hypersphere NEF n=5000):

  | dataset | NEF baseline | best MLP | best MLP cell | Δ |
  |---|---|---|---|---|
  | MNIST | 97.08 ± 0.04 | **98.10 ± 0.08** | d1_w512 lr=3e-3 (407 k, 22 ep, 58 s) | +1.0 |
  | Fashion | 87.75 ± 0.06 | **89.39 ± 0.08** | d2_w1024 lr=1e-3 (1.86 M, 24 ep, 254 s) | +1.6 |
  | CIFAR-10 | 49.72 ± 0.08 | **53.73 ± 0.22** | d2_w1024 lr=1e-3 (4.21 M, 34 ep, 327 s) | +4.0 |

- Honest framing: a properly tuned gradient-trained MLP **beats**
  hypersphere NEF on all three datasets, by a margin that grows
  with task difficulty.  We do not present this as a NEF-MLP
  contest; we present it as the gradient-trained reference that
  any reader expects to see.
- **The interesting twist**: B3 receptive_field NEF on CIFAR-10
  reaches **55.10 ± 0.10**, beating the best B4 MLP (53.73 ± 0.22)
  by **+1.37 pp** at well outside seed noise.  When the encoder
  carries the spatial prior, a fully gradient-free single-layer
  NEF *exceeds* a 4.2 M-parameter gradient-trained MLP on CIFAR-10,
  in a ~25 s analytic solve vs a ~5 min Adam fit.
- **The story**: the encoder, not the decoder, is where
  representational work happens.  A good encoder closes (and
  sometimes reverses) the single-layer / gradient-trained gap.
  This is the right framing for the paper — it points naturally
  at Paper II (continual learning, where the analytic-solve
  property *is* the win) and Paper III (RL, where the same
  accumulator handles harder targets).
- What this is **not**: a claim that NEF beats deep convnets, or
  that NEF is universally faster than gradient training.  Tuned
  MLPs beat hypersphere NEF; a good prior closes the gap on
  spatial data; deeper gradient-trained models would extend the
  gap again.  We state all three observations side-by-side.

## §8 — What is novel, what is replication, what we honestly claim  [insights §9]

- Novelty: the frame-coefficient pedagogical framing; the trace-
  scaled α footgun-prevention story; the streaming-NEF /
  multi-seed reproducibility infrastructure.
- Replication: ELM (Huang et al. 2006), random features (Rahimi &
  Recht 2007), Coates et al. 2011 (k-means features), reservoir
  computing (Jaeger 2001).
- No claim-pumping and no hiding failures; separate replication from
  genuine contribution.

## §9 — Discussion and forward looks

- Calibrated uncertainty for free from `(AᵀA + αI)⁻¹` (one
  paragraph; deferred work).
- Continual learning: per-class additivity of `AᵀA` and `AᵀY`
  → no catastrophic forgetting.  See Paper II.
- RL: same accumulator, harder targets.  See Paper III.

## Appendix A — Reproducibility

- Code identity hash, seed list, hardware, exact config dicts.
- Every JSON in `results/sl_clean/` is a verifiable claim.

## Appendix B — Negative results (kept honest)  [insights §8]

- Multi-layer gradient-free strategies: target-prop, hybrid, e2e all
  match a fully-trained MLP only when the last layer reverts to
  e2e SGD; the analytic-solve property does not extend through
  stacked nonlinearities.  We document this and stop.
- Recurrent NEF (state feedback through fixed encoders) compounds
  state-feedback noise across timesteps.  E2E gradient training
  is what makes RecurrentNEFLayer competitive; greedy and target-
  prop variants underperform.  Documented; not the path forward.

---

## Open design questions

1. Should §1 (six principles) be its own short section, or merged
   into the abstract?  Currently planned as a section so all three
   papers have an identifiable shared anchor.
2. Does §2 (over-representation needs nonlinearity) belong before
   or after §3 (method)?  Outline currently puts the geometry
   first, then the implementation.  Alternative: implementation
   first as a "what we do", geometry second as "why it works".
3. How aggressive to be about the H8 linear-collapse sanity-check?
   Currently planned as analytical-only (no experiment); could
   add a 5-min experiment if H8 sceptics surface in review.

---

## Figure list (draft)

Pencilled out so we know what data we still need; numbering is
sectional and will be normalized during synthesis (Phase D).

- **F1 (§1).** Conceptual diagram: the six principles as a
  dependency graph (over-representation → nonlinearity → frame
  coefficients → analytic decoder → α scales with span → data-
  driven biases as fold placement).  Hand-drawn; no data.
- **F2 (§2).** Frame-coefficient cartoon: 2-D toy showing
  `aᵢ = αᵢ |⟨eᵢ, x − dᵢ⟩|` for a small ensemble of fold
  hyperplanes, with abs producing a piecewise-linear tessellation.
  Generated; ~200 LOC.
- **F3 (§4).** α-as-function-of-n curve.  X = n_neurons (log),
  Y = test accuracy, families = α policy ∈ {fixed-1e-2,
  fixed-1e-4, trace+1e-2, trace+1e-4, trace+1e-3}.  Source:
  `results/sl_clean/streaming_alpha_sweep_*.json`.  Shows
  trace+1e-4 wins worst-case across n; fixed-1e-2 collapses at
  n=8k.
- **F4 (§5.1).** B1 panel — abs hinge geometry.  3 datasets ×
  bias_mode {data_driven, zero, random_normal} × encoder
  {hypersphere, gaussian, sparse}.  Mean ± SE bars.  Source:
  `results/sl_clean/biases_{mnist,fashion,cifar10}_panel.json`.
- **F5 (§5.1).** B7 falsifier — centered CIFAR-10 + decentered MNIST
  re-open / reverse the gap.  4 conditions side-by-side with the
  "expected if pure-DC" baseline marked.  Source:
  `results/sl_clean/biases_{cifar10_centered,mnist_decentered}_panel.json`.
- **F6 (§5.x).** B3 encoder-strategy panel.  3 datasets × 7
  strategies × 10 seeds.  Highlights: receptive_field +5.4 pp on
  CIFAR-10; class_contrast clean failure.  Source:
  `results/sl_clean/encoders_{mnist,fashion,cifar10}_panel.json`.
- **F7 (§6.2).** Stream-NEF row sMNIST: accuracy vs n (2k/4k/8k)
  for three solve modes (batch_acc / accumulate / Woodbury) with
  H3 equivalence overlaid.  Source: `results/sl_clean/b2_*.json`.
- **F8 (§6.3).** Stream-NEF permuted sMNIST: accuracy vs n with
  ESN ceiling (B6) and LSTM single-seed point (B5) overlaid.
  Source: `results/sl_clean/{b2,b5_lstm,b6_esn}_*.json`.
- **F9 (§7).** B4 single-layer comparator: best-cell MLP vs B1
  hypersphere NEF vs B3 RF NEF on 3 datasets.  Bar chart with CIs;
  parameter counts annotated.  Source:
  `results/sl_clean/{b1,b3_encoders,b4_mlp}_*.json`.
- **F10 (§8 / Appendix).** Cost-vs-quality scatter: x=wall_sec
  (log), y=test acc.  All single-layer / streaming methods
  plotted; gradient-free vs gradient-trained marked by shape.
  Souces: aggregated.

Open: a frame-coefficient *measured* figure (vs the §2 cartoon) —
empirical activity histogram on a real dataset showing the
piecewise-linear fold structure.  Deferred until we have a clear
visualisation that adds something the cartoon does not.
