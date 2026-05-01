# Paper I outline (Supervised Learning, ending with Stream-NEF)

This is the **structural skeleton** that the Paper I text will follow.
Numbers in `[…]` reference sections of `paper_insights.md` where the
content lives now.  The text-writing pass will turn each bullet into a
paragraph; we are not writing prose here.

The trilogy's shared opener (the six principles, §0pre of insights)
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

### §5.1 — DC component and dataset centeredness  [insights §1.2]

- Setup: NEFLayer with biases ∈ {data_driven, zero}.  With abs
  activation, the bias is the fold point of `aᵢ(x) = |α·eᵢ·x − b|`.
- For centred datasets (MNIST, Fashion), the dataset has DC ≈ 0;
  zero biases place the fold at the cloud's mean → identical to
  data-driven biases.  Empirically <0.2 pp difference (B1 result).
- For non-centred datasets (CIFAR-10), the dataset has a strong DC
  component; zero biases place the fold off-cloud, where `|·|`
  acts almost-linearly.  Data-driven biases place the fold inside
  the cloud where the nonlinearity actually does work.
- **Predicted experiment**: per-channel-centre CIFAR-10 → the gap
  between zero and data_driven *closes* (b1-followup-cifar-centered).
  Conversely, de-centred MNIST → the gap *appears*
  (b1-followup-mnist-decentered).  Both queued.

### §5.2 — Encoder strategy is mostly orthogonal to bias choice  [insights §6]

- 7 strategies (hypersphere, gaussian, sparse, receptive_field,
  whitened, class_contrast, local_pca).
- B1 result: with data-driven biases, all 7 strategies cluster
  within ~0.5 pp on MNIST and Fashion.  Receptive_field wins by a
  hair on CIFAR-10 only because it embeds spatial locality, not
  because of analytic-solve behaviour.
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

- 10 seeds × {n=2k, 4k, 8k} (B2 final panel).
- Honest LSTM comparison: cite published values; report whatever
  we measure.  No headline pumping.
- Wall-clock comparison **labelled CPU** unless we add GPU runs.

### §6.3 — Permuted sMNIST results

- 5 seeds × {n=2k, 4k} (B2 final panel).  T=784 vs T=28; honest
  scope statement.

### §6.4 — Limits  [insights §4]

- Window K bounds the receptive field; periodicities longer than K
  are invisible to mean pooling.
- For non-stationary sequences (audio, video), mean-pool washes out
  per-position information.  This is the boundary where pooling
  stops carrying us and recurrence becomes necessary — the bridge
  to Paper III.

## §7 — What is novel, what is replication, what we honestly claim  [insights §9]

- Novelty: the frame-coefficient pedagogical framing; the trace-
  scaled α footgun-prevention story; the streaming-NEF /
  multi-seed reproducibility infrastructure.
- Replication: ELM (Huang et al. 2006), random features (Rahimi &
  Recht 2007), Coates et al. 2011 (k-means features), reservoir
  computing (Jaeger 2001).
- We are not in the science business.  No claim-pumping.

## §8 — Discussion and forward looks

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
