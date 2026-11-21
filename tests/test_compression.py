import math
from dataclasses import replace

import pytest
import torch

from benchmarks.compression import series
from benchmarks.compression.baselines import AdaptiveCategorical, ARConfig, PrequentialAR
from benchmarks.compression.prequential import PredictorConfig, score_symbols
from benchmarks.compression.quantization import UniformQuantizer
from leenef import NEFLayer
from leenef.solvers import inverse_from_normal_equations


def test_quantizer_clips_and_roundtrips_symbols():
    grid = UniformQuantizer(lo=-1.5, hi=1.5, n_bits=2)
    values = torch.tensor([-1e300, -1.5, -1.0, 0.0, 1.0, 1.5, 1e300], dtype=torch.float64)
    assert grid.quantize(values).tolist() == [0, 0, 0, 2, 2, 3, 3]
    symbols = torch.arange(4)
    torch.testing.assert_close(grid.quantize(grid.reconstruct(symbols)), symbols)
    wide_grid = UniformQuantizer(-10, 3, 32)
    symbols = torch.tensor([0, 1, 2**31, 2**32 - 2, 2**32 - 1])
    torch.testing.assert_close(wide_grid.quantize(wide_grid.reconstruct(symbols)), symbols)


def test_quantizer_rejects_invalid_inputs():
    for bounds in ((1, 1), (2, 1), (0, math.inf), (-1e308, 1e308), (1e16, 1e16 + 2)):
        with pytest.raises(ValueError):
            UniformQuantizer(*bounds, n_bits=2)
    for depth in (0, 33, True):
        with pytest.raises(ValueError, match="n_bits"):
            UniformQuantizer(-1, 1, depth)
    grid = UniformQuantizer(-1, 1, 2)
    with pytest.raises(ValueError, match="finite"):
        grid.quantize(torch.tensor([math.nan]))
    with pytest.raises(ValueError, match="outside"):
        grid.reconstruct(torch.tensor([-1, 4]))
    with pytest.raises(ValueError, match="integers"):
        grid.reconstruct(torch.tensor([0.7]))


def test_gaussian_cells_match_cdf_and_form_normalized_alphabet():
    grid = UniformQuantizer(0, 3, 2)
    models = torch.tensor([[0.0], [1.4], [3.0]], dtype=torch.float64)
    variance = torch.tensor([[1.0], [0.37], [0.2]], dtype=torch.float64)
    bounds = torch.tensor([-math.inf, 0.5, 1.5, 2.5, math.inf], dtype=torch.float64)
    cdf = 0.5 * (1 + torch.erf((bounds - models) / torch.sqrt(2 * variance)))
    actual = grid.gaussian_log_prob(torch.arange(4), models, variance).exp()
    torch.testing.assert_close(actual, cdf.diff(), atol=1e-15, rtol=1e-12)
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(3, dtype=torch.float64))


def test_gaussian_cells_have_finite_tails_and_correct_scale_limits():
    grid = UniformQuantizer(0, 3, 2)
    # Both left and right tails, including an interior cell and saturated cells.
    means = torch.tensor([40.5, -39.5, 42.5, -37.5], dtype=torch.float64)
    expected = 804.6084420137538 / math.log(2)
    bits = grid.gaussian_bits(torch.arange(4), means, torch.tensor(1.0))
    torch.testing.assert_close(bits, torch.full_like(bits, expected), rtol=1e-13, atol=0)

    tiny = torch.tensor(1e-200, dtype=torch.float64)
    assert grid.gaussian_bits(torch.tensor(1), torch.tensor(1.0), tiny) == 0
    assert grid.gaussian_bits(torch.tensor(1), torch.tensor(0.5), tiny) == pytest.approx(1)

    broad = torch.tensor(1e300, dtype=torch.float64)
    bits = grid.gaussian_bits(torch.arange(4), torch.tensor(0.0), broad)
    interior = math.log2(math.sqrt(2 * math.pi) * 1e150)
    assert bits.tolist() == pytest.approx([1, interior, interior, 1], abs=1e-12)


def test_gaussian_cells_reject_invalid_parameters():
    grid = UniformQuantizer(0, 3, 2)
    for variance in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError, match="variances"):
            grid.gaussian_bits(torch.tensor(1), torch.tensor(0.0), torch.tensor(variance))
    with pytest.raises(ValueError, match="means"):
        grid.gaussian_bits(torch.tensor(1), torch.tensor(math.nan), torch.tensor(1.0))
    with pytest.raises(FloatingPointError, match="representable"):
        grid.gaussian_bits(
            torch.tensor(1), torch.tensor(1e300, dtype=torch.float64), torch.tensor(1.0)
        )


@pytest.mark.parametrize(
    ("generate", "expected"),
    [
        (series.mackey_glass, [0.9011587866733866, 0.8321311055260482, 0.7647495464948321]),
        (series.lorenz, [1.2860891062830582, 1.729007793440235, 2.391614474119744]),
        (series.henon, [-0.741620375487425, 0.5582529857066737, 0.35228572312027967]),
        (series.rossler, [0.09127308485989319, 0.09472605139765196, 0.08923297554222852]),
    ],
)
def test_generators_match_source_before_standardization_and_isolate_rng(generate, expected):
    before = torch.random.get_rng_state()
    values = generate(8, warmup=4, seed=7)
    torch.testing.assert_close(
        values[:3], torch.tensor(expected, dtype=torch.float64), atol=2e-15, rtol=0
    )
    torch.testing.assert_close(values, generate(8, warmup=4, seed=7), atol=0, rtol=0)
    assert not torch.equal(values, generate(8, warmup=4, seed=8))
    assert torch.equal(before, torch.random.get_rng_state())


def test_generator_rejects_invalid_arguments_and_divergence():
    with pytest.raises(ValueError, match="nonfinite observations"):
        series.henon(8, warmup=0, seed=1, a=1e300, noise_std=0)
    with pytest.raises(ValueError, match="noise_std"):
        series.rossler(2, warmup=1, seed=1, noise_std=-0.01)


@pytest.fixture
def stream():
    grid = UniformQuantizer(-1, 1, 4)
    config = PredictorConfig(
        window=2,
        width=5,
        warmup=3,
        resolve_every=3,
        fixed_alpha=0.2,
        noise_variance=0.3,
        seed=7,
        activation="gelu",
        refresh_every=2,
    )
    symbols = torch.tensor([0, 4, 9, 5, 11, 2, 14, 8, 1, 9, 4, 5, 15, 11, 2, 0])
    return grid, config, symbols


def test_prequential_matches_direct_source_refit_timing_and_moments(stream):
    grid, config, symbols = stream
    result = score_symbols(symbols, grid, config)
    assert result.uniform_prefix == config.window + config.warmup
    assert result.bits[: result.uniform_prefix].tolist() == [4] * result.uniform_prefix
    assert result.total_bits == pytest.approx(sum(result.bits.tolist()))

    # Independent direct-prefix reference to the fixed-parameter source expert.
    recon = grid.lo + symbols.double() * grid.delta
    windows = recon[:-1].float().unfold(0, config.window, 1).contiguous()
    rng = torch.Generator().manual_seed(config.seed)
    centers = windows[: config.warmup][torch.randperm(config.warmup, generator=rng)]
    layer = NEFLayer(config.window, config.width, 1, "gelu", rng=rng, centers=centers)
    phi = layer.encode(windows).double()
    y = recon[config.window :]
    published = (3, 4, 7, 10, 13)
    for i in range(config.warmup, y.numel()):
        rows = max(n for n in published if n <= i)
        prefix = phi[:rows]
        inverse = inverse_from_normal_equations(prefix.T @ prefix, alpha=(0.2, "fixed"))
        p = phi[i]
        mean = float(p @ (inverse @ (prefix.T @ y[:rows])))
        var = 0.3 * (1 + float(p @ inverse @ p))
        t = i + config.window
        j = t - result.uniform_prefix
        assert float(result.mean[j]) == pytest.approx(mean, abs=2e-7)
        assert float(result.variance[j]) == pytest.approx(var, abs=2e-7)
        s = int(symbols[t])
        center = float(y[i])
        denom = math.sqrt(2 * var)
        below = -1 if s == 0 else math.erf((center - grid.delta / 2 - mean) / denom)
        above = 1 if s == grid.last_symbol else math.erf((center + grid.delta / 2 - mean) / denom)
        assert float(result.bits[t]) == pytest.approx(-math.log2((above - below) / 2), abs=2e-7)


def test_prequential_predictions_are_causal_and_reproducible(stream):
    grid, config, symbols = stream
    before = torch.random.get_rng_state()
    result = score_symbols(symbols, grid, config)
    duplicate = score_symbols(symbols, grid, config)
    for a, b in zip(
        (result.bits, result.mean, result.variance),
        (duplicate.bits, duplicate.mean, duplicate.variance),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    different_features = score_symbols(symbols, grid, replace(config, seed=config.seed + 1))
    assert not torch.equal(result.mean, different_features.mean)
    assert torch.equal(before, torch.random.get_rng_state())

    for t in (5, 6, 9, 12):
        changed = symbols.clone()
        changed[t:] = grid.last_symbol - changed[t:]
        alternative = score_symbols(changed, grid, config)
        shortened = score_symbols(symbols[: t + 1], grid, config)
        j = t - result.uniform_prefix
        for other in (alternative, shortened):
            torch.testing.assert_close(result.mean[: j + 1], other.mean[: j + 1], atol=0, rtol=0)
            torch.testing.assert_close(
                result.variance[: j + 1], other.variance[: j + 1], atol=0, rtol=0
            )
            torch.testing.assert_close(result.bits[:t], other.bits[:t], atol=0, rtol=0)
        assert alternative.bits[t] != result.bits[t]

    prefix_only = score_symbols(symbols[:4], grid, config)
    assert prefix_only.total_bits == 4 * grid.n_bits
    assert prefix_only.mean.numel() == prefix_only.variance.numel() == 0


def test_prequential_does_not_fit_raw_values_or_accept_invalid_config(stream):
    grid, config, symbols = stream
    middle = symbols[1:-2].clamp(1, grid.last_symbol - 1)
    centers = grid.reconstruct(middle)
    a = score_symbols(grid.quantize(centers), grid, config)
    b = score_symbols(grid.quantize(centers + grid.delta / 3), grid, config)
    torch.testing.assert_close(a.bits, b.bits, atol=0, rtol=0)
    with pytest.raises(ValueError, match="fixed_alpha"):
        replace(config, fixed_alpha=math.nan)
    with pytest.raises(ValueError, match="noise_variance"):
        replace(config, noise_variance=-1)


def test_categorical_predictive_probabilities_and_updates():
    grid = UniformQuantizer(0, 3, 2)
    coder = AdaptiveCategorical(grid, pseudocount=0.5)
    assert [coder.bits_for(s) for s in range(4)] == [2.0] * 4
    assert coder.score_symbols(torch.tensor([1, 1, 3])).tolist() == pytest.approx([2.0, 1.0, 3.0])
    probabilities = [2 ** -coder.bits_for(s) for s in range(4)]
    assert probabilities == pytest.approx([0.1, 0.5, 0.1, 0.3])
    assert sum(probabilities) == pytest.approx(1.0)

    suffix = torch.tensor([1, 0, 0])
    other = AdaptiveCategorical(grid, pseudocount=0.5)
    batch = other.score_symbols(torch.cat((torch.tensor([1, 1, 3]), suffix)))
    torch.testing.assert_close(coder.score_symbols(suffix), batch[3:], atol=0, rtol=0)
    changed = AdaptiveCategorical(grid, pseudocount=0.5)
    prefix = changed.score_symbols(torch.tensor([1, 1, 3, 2]))
    torch.testing.assert_close(prefix[:3], batch[:3], atol=0, rtol=0)
    assert prefix[3] == pytest.approx(-math.log2(probabilities[2]))


def test_categorical_large_sparse_alphabet_and_extreme_pseudocounts():
    grid = UniformQuantizer(0, 3, 32)
    k = 2**32
    for a in (0.5, 1e308, math.ulp(0.0)):
        coder = AdaptiveCategorical(grid, pseudocount=a)
        bits = coder.score_symbols(torch.tensor([k - 1, k - 1]))
        assert bits[0] == 32
        if a == 0.5:
            assert bits[1] == pytest.approx(math.log2((1 + k * a) / (1 + a)))
            assert coder.bits_for(0) == pytest.approx(math.log2((2 + k * a) / a))
        elif a > 1:
            assert bits[1] == pytest.approx(32)
            assert coder.bits_for(0) == pytest.approx(32)
        else:
            assert 0 < bits[1] < 1e-300
            assert coder.bits_for(0) == pytest.approx(1075)
        seen, unseen = (coder.bits_for(s) for s in (k - 1, 0))
        assert 2**-seen + (k - 1) * 2**-unseen == pytest.approx(1.0)


def test_ar_matches_closed_form_refits_and_scores_before_update():
    grid = UniformQuantizer(0, 3, 2)
    config = ARConfig(order=1, warmup=2, resolve_every=3, fixed_alpha=2.0, noise_variance=0.25)
    seq = [0, 1, 2, 0, 3, 1, 3, 2, 3, 0, 3, 2]
    result = PrequentialAR(grid, config).score_symbols(torch.tensor(seq))
    assert result.uniform_prefix == 3
    assert result.bits[:3].tolist() == [2.0] * 3
    for i, (x, y) in enumerate(zip(seq[:-1], seq[1:], strict=True)):
        if i < 2:
            continue
        rows = max(count for count in (2, 4, 7, 10) if count <= i)
        xs, ys = seq[:rows], seq[1 : rows + 1]
        a, b, d = rows + 2, sum(xs), sum(p * p for p in xs) + 2
        u = sum(ys)
        v = sum(p * q for p, q in zip(xs, ys, strict=True))
        determinant = a * d - b**2
        mean = (d * u - b * v + x * (a * v - b * u)) / determinant
        variance = 0.25 * (1 + (d - 2 * b * x + a * x**2) / determinant)
        assert float(result.mean[i - 2]) == pytest.approx(mean)
        assert float(result.variance[i - 2]) == pytest.approx(variance)
        scale = math.sqrt(2 * variance)
        lower = -1 if y == 0 else math.erf((y - 0.5 - mean) / scale)
        upper = 1 if y == 3 else math.erf((y + 0.5 - mean) / scale)
        assert float(result.bits[i + 1]) == pytest.approx(-math.log2((upper - lower) / 2))


def test_ar_chunk_parity_causality_and_reconstruction():
    grid = UniformQuantizer(-1.5, 1.5, 2)
    config = ARConfig(order=2, warmup=2, resolve_every=3, fixed_alpha=0.1, noise_variance=0.2)
    symbols = torch.tensor([1, 2, 2, 1, 1, 0, 2, 3, 0])
    coder = PrequentialAR(grid, config)
    result = PrequentialAR(grid, config).score_symbols(symbols)
    chunks = [coder.score_symbols(part) for part in symbols.split((0, 3, 1, 2, 3))]
    assert [part.uniform_prefix for part in chunks] == [0, 3, 1, 0, 0]
    for field in ("bits", "mean", "variance"):
        torch.testing.assert_close(
            torch.cat([getattr(part, field) for part in chunks]),
            getattr(result, field),
            atol=0,
            rtol=0,
        )

    changed = symbols.clone()
    changed[5:] = 3 - changed[5:]
    other = PrequentialAR(grid, config).score_symbols(changed)
    torch.testing.assert_close(result.bits[:5], other.bits[:5], atol=0, rtol=0)
    for field in ("mean", "variance"):
        torch.testing.assert_close(
            getattr(result, field)[:2], getattr(other, field)[:2], atol=0, rtol=0
        )
    assert result.bits[5] != other.bits[5]
    centers = grid.reconstruct(symbols)
    near_centers = grid.quantize(centers + grid.delta / 3)
    reconstructed = PrequentialAR(grid, config).score_symbols(near_centers)
    torch.testing.assert_close(result.bits, reconstructed.bits, atol=0, rtol=0)


def test_compression_baselines_stable_and_do_not_draw_global_rng():
    grid = UniformQuantizer(-1, 1, 2)
    # The initial design is rank one for five coefficients. Use an unlikely
    # saturated cell to test the log-domain score far out in the left tail.
    config = ARConfig(order=4, warmup=1, resolve_every=2, fixed_alpha=0.1, noise_variance=1e-4)
    symbols = torch.tensor([3, 3, 3, 3, 3, 0, 3, 3, 3])
    before = torch.random.get_rng_state()
    result = PrequentialAR(grid, config).score_symbols(symbols)
    AdaptiveCategorical(grid, pseudocount=0.5).score_symbols(symbols)
    assert torch.equal(before, torch.random.get_rng_state())
    assert float(result.mean[0]) == pytest.approx(5 / 5.1)
    assert float(result.variance[0]) == pytest.approx(1e-4 * (1 + 5 / 5.1))
    assert torch.isfinite(result.bits).all()
    assert result.bits[5] > 1000


def test_compression_baselines_reject_invalid_inputs():
    grid = UniformQuantizer(0, 3, 2)
    config = ARConfig(order=1, warmup=2, resolve_every=3, fixed_alpha=2.0, noise_variance=0.25)
    for name in ("order", "warmup", "resolve_every"):
        for bad in (0, True, 1.5):
            with pytest.raises(ValueError, match=name):
                replace(config, **{name: bad})
    for bad in (0, -1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="pseudocount"):
            AdaptiveCategorical(grid, pseudocount=bad)
        for name in ("fixed_alpha", "noise_variance"):
            with pytest.raises(ValueError, match=name):
                replace(config, **{name: bad})
    categorical = AdaptiveCategorical(grid, pseudocount=0.5)
    ar = PrequentialAR(grid, config)
    ar.score_symbols(torch.tensor([0, 1, 2]))
    for scorer in (categorical, ar):
        with pytest.raises(ValueError, match="one-dimensional CPU"):
            scorer.score_symbols(torch.tensor([[1, 2]]))
        with pytest.raises(ValueError, match="integers"):
            scorer.score_symbols(torch.tensor([1.0]))
        with pytest.raises(ValueError, match="alphabet"):
            scorer.score_symbols(torch.tensor([1, 4]))
    assert categorical.bits_for(1) == 2.0  # Invalid batch did not count its valid prefix.
    with pytest.raises(ValueError, match="alphabet"):
        categorical.bits_for(-1)
    expected = PrequentialAR(grid, config).score_symbols(torch.tensor([0, 1, 2, 0, 3]))
    torch.testing.assert_close(
        ar.score_symbols(torch.tensor([0, 3])).bits, expected.bits[3:], atol=0, rtol=0
    )


def test_ar_refit_failure_makes_scorer_unusable():
    wide = UniformQuantizer(-1e155, 1e155, 2)
    config = ARConfig(order=1, warmup=2, resolve_every=3, fixed_alpha=2.0, noise_variance=0.25)
    ar = PrequentialAR(wide, config)
    with pytest.raises(FloatingPointError, match="sufficient statistics"):
        ar.score_symbols(torch.tensor([3, 3, 3, 3]))
    with pytest.raises(RuntimeError, match="unusable.*fresh scorer"):
        ar.score_symbols(torch.tensor([0]))


def test_ar_predictive_overflow_makes_scorer_unusable():
    grid = UniformQuantizer(-1, 1, 2)
    config = ARConfig(order=1, warmup=1, resolve_every=1, fixed_alpha=1.0, noise_variance=1.5e308)
    ar = PrequentialAR(grid, config)
    with pytest.raises(FloatingPointError, match="nonfinite AR predictive moments"):
        ar.score_symbols(torch.tensor([3, 3, 3]))
    with pytest.raises(RuntimeError, match="unusable.*fresh scorer"):
        ar.score_symbols(torch.empty(0, dtype=torch.int64))
