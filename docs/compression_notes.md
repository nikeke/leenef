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
