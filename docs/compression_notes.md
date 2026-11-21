# C0: ideal prequential codelength

`benchmarks/compression/` adapts the single-expert leenef experiment from
the author's local `common/compression/solomonoff_leenef` project:
`series.py` for trajectories, `codec.py` for the grid, and
`compress.py`, `gauss_code.py`, and `evidence.py` for the score/update
schedule and Gaussian predictor. The port has no runtime dependency on
that project. It excludes prior/mixture machinery, model selection,
baseline codecs, and growth or pruning.

- `series.py` returns raw float64 Mackey–Glass, Lorenz, Hénon, and
  Rössler observations. It preserves the source's integration/map,
  initial conditions, burn-in, subsampling, noise defaults, and draw
  order. In particular, Mackey–Glass draws its initial history before
  observation noise from the same local generator. The generator and
  predictor have independent, explicitly seeded RNGs. A generated
  trajectory is reproducible for the same arguments and PyTorch
  environment; changing `n` can change PyTorch's batched noise draws
  even in the common prefix. No full-trajectory standardization is used.
- `UniformQuantizer(lo, hi, n_bits)` takes fixed bounds as side
  information, e.g. from a known source or a separate, earlier
  calibration segment. The local capstone instead used full-stream
  extrema and counted a header; that is not a range an online encoder
  can know in advance. Do not derive bounds or any external
  normalization from the stream being scored. Values outside the
  range saturate, ties round to even, and reconstruction returns a
  bin center, not the original observation. The grid rejects
  degenerate or unresolvable ranges and supports 1–32 bits.
- `score_symbols(symbols, grid, config)` consumes a one-dimensional
  CPU symbol sequence. The first `window + warmup` symbols (or the
  entire stream if shorter) use `n_bits` each. The source always
  scores the initial window uniformly, but defaults to an adaptive
  order-0 marginal coder for the subsequent warmup targets. C0's
  uniform warmup targets match the source's `--no-marginal` mode, not
  its default. Feature centers come only from the reconstructed warmup
  windows, with the source's seeded candidate sampling and
  gain/encoder setup. For each remaining
  symbol, its window contains only previous bin centers. Score it
  before adding its reconstructed target to the pending fit data.
  Like the source, publish a fit after the warmup targets and then
  after zero-based **target-window** indices divisible by
  `resolve_every`; intermediate predictions use the last published
  fit, not the pending observations. Center sampling, encoder
  generation, and gain generation share the predictor's RNG as in
  the source: changing warmup or width may change subsequent draws
  even with the same seed.

The source reselected per-expert precisions at each refit. C0 instead
requires a fixed positive `fixed_alpha` and `noise_variance` specified
before scoring. These correspond to source precisions
`fixed_alpha = alpha_p / beta` and `noise_variance = 1 / beta`. The
single-expert mean and variance use the same formulas at fixed
precisions: with `M = A.T @ A + fixed_alpha * I`, the decoder is
`M^-1 @ A.T @ y` and the variance is
`noise_variance * (1 + phi @ M^-1 @ phi.T)`. This uses the centralized
fixed ridge policy, not trace scaling: a cached Woodbury inverse cannot
change alpha midstream. The fit builds the inverse from float64
sufficient statistics at warmup, on large updates, and periodically
(`refresh_every`, in refits); intervening refits use batched Woodbury.
Sequential float32 feature evaluation and batched float64 math can
differ in rounding from the source's batched feature evaluation and
eigensolves. The source comparison is limited to one expert at fixed
precisions with `--no-marginal` warmup; it does not reproduce the
source's default configuration or mixture results.

The reported total sums **ideal estimated bits**, `-log2 P(symbol)`.
Gaussian cells absorb the outer tails; probabilities use log-CDF
differences and a narrow-cell quadrature to avoid the source's
probability floor. Invalid inputs and nonrepresentable codelengths
raise rather than silently flooring probabilities. Totals include the
uniform prefix but omit configuration, calibration, headers, and any
entropy-coder overhead. There is no serialized bitstream, measured byte
count, or lossless decoding claim. C0 contains no benchmark results.

## C1: categorical and linear controls

`benchmarks/compression/baselines.py` accepts CPU integer symbol vectors
on the same externally fixed `UniformQuantizer` as C0. Construct a new
scorer for each independent stream. Each `score_symbols` call scores and
consumes its input in order; splitting a stream into chunks does not
change its scores or refit schedule. Neither baseline draws randomness.

- `AdaptiveCategorical(grid, pseudocount=a)` uses the finite-alphabet
  prediction $P_t(s)=(c_t(s)+a)/(t+Ka)$, where $K=2^{\mathrm{n\_bits}}$,
  $c_t(s)$ counts occurrences among the first $t$ symbols, and $a$ is
  fixed, finite, and positive. The prior includes **all** grid cells
  even though storage grows only with observed distinct symbols.
  `bits_for(s)` inspects the current ideal bits without updating.
  The first prediction is uniform (`n_bits`); every subsequent
  prediction uses the counts so far. Unlike C0, this model needs no
  lag or warmup, so it adapts even on symbols for which C0 pays
  `n_bits`. It does not restart at the common suffix boundary.
- `PrequentialAR(grid, ARConfig(...))` uses exactly `order` previous
  bin centers, oldest first, plus a constant intercept to represent a
  nonzero level. In float64 it accumulates the normal equations over
  all observed lag/target pairs, using bin centers for both inputs
  and targets. It includes the intercept in the fixed ridge penalty,
  so the same regularization makes the system positive definite even
  with fewer warmup rows than coefficients. With lag row $\phi$ and
  $M=A^\top A+\alpha_{\mathrm{fixed}}I$, it predicts
  $\phi M^{-1}A^\top y$ and uses C0's Gaussian cells with variance
  $\sigma^2(1+\phi M^{-1}\phi^\top)$. The observation variance
  `noise_variance` ($\sigma^2$) is fixed in squared grid-value units;
  only the leverage term changes, not the noise scale. Each refit
  recomputes the inverse using the centralized fixed ridge policy.
  The adjacent source's train/test AR reference uses residual-based
  calibration; this control does not reproduce it.

AR requires positive integer `order`, `warmup`, and `resolve_every`,
and finite positive `fixed_alpha` and `noise_variance`. It charges
`n_bits` for the first `order + warmup` symbols (or the whole stream
if shorter). It publishes its initial fit after consuming the `warmup`-th
target, then refits after consuming targets with zero-based target-window
index $i$ divisible by `resolve_every` (target position `order + i`).
The current symbol is scored before the update, and between refits
predictions use the last published fit. This is causal updating on one
stream, not a train/validation/test fit or a procedure for selecting
any parameter. No statistics from the scored stream set the grid,
noise scale, ridge strength, or other configuration.

AR validates the entire chunk before updating; invalid input does not
alter the scorer. A numerical failure while scoring or refitting aborts the
chunk without returning scores and makes that scorer unusable. Subsequent
calls raise `RuntimeError`; create a fresh scorer and replay from the
start instead of retrying on the failed instance.

For later comparisons, report **both** all-stream bits (`bits.sum()`)
and common-suffix bits (`bits[start:].sum()`) for every method on the
same symbols, with `start = max(C0 window + warmup, AR order + warmup)`
and numerator/length (or bits per symbol for nonempty segments).
Categorical `score_symbols` returns the bits tensor; Gaussian scorers
return it as `result.bits` plus their moments on the modeled suffix.
All-stream bits include each method's actual prefix cost. These are
ideal lengths only and omit prior/pseudocount specification, calibration,
configuration, headers, and entropy-coder overhead; they are not
measured bytes or a bit-exact decoding demonstration.

One candidate for parent review, not a registered configuration or
scientific gate: Hénon with `n=512`, trajectory seeds `0, 1, 2`,
`a=1.4`, `b=0.3`, burn-in `200`, and noise standard deviation `0.005`;
fix the grid in advance to `[-2, 2]` with 8 bits. Try `pseudocount=0.5` per cell
for categorical smoothing. Match AR order and C0 window at 2, warmup
at 32 targets, and `resolve_every=8`; explicitly fix both
`fixed_alpha=0.1` and `noise_variance=0.01`. For C0 use `abs`, width
16, predictor seed 7, and `refresh_every=16`. Share the input symbols,
record saturation counts without changing the bounds, and report
all-stream and common-suffix scores per trajectory. The same ridge
coefficient does not imply equal shrinkage across feature maps; a
single fixed configuration is a control, not evidence of optimality
or a complete matched comparison. This foundation runs no panel and
contains no benchmark results.
