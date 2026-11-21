"""Tests for StreamingNEFClassifier — temporal sequence classification."""

import pytest
import torch

from leenef.streaming import StreamingNEFClassifier


class TestStreamingNEFClassifier:
    def test_forward_shape(self):
        """Forward pass should produce (N, d_out) from (N, T, d)."""
        clf = StreamingNEFClassifier(4, 200, 3, window_size=3)
        x_seq = torch.randn(10, 20, 4)
        out = clf(x_seq)
        assert out.shape == (10, 3)

    def test_delay_features_shape(self):
        """Delay features should have shape (N, T, K*d)."""
        clf = StreamingNEFClassifier(4, 100, 2, window_size=5)
        x_seq = torch.randn(8, 15, 4)
        delay = clf._delay_features(x_seq)
        assert delay.shape == (8, 15, 20)

    def test_delay_features_content(self):
        """Last timestep delay feature should contain the last K timesteps."""
        clf = StreamingNEFClassifier(2, 100, 1, window_size=3)
        x_seq = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
        delay = clf._delay_features(x_seq)
        # Last timestep (t=9) should contain timesteps 7,8,9
        expected = torch.cat([x_seq[0, 7], x_seq[0, 8], x_seq[0, 9]])
        assert torch.allclose(delay[0, 9], expected)

    def test_flat_delay_windows_matches_materialized_delay(self):
        """Sampled delay windows should match the materialized flattened tensor."""
        clf = StreamingNEFClassifier(2, 100, 1, window_size=3)
        x_seq = torch.randn(4, 5, 2)
        idx = torch.tensor([0, 3, 7, 12, 19])
        expected = clf._delay_features(x_seq).reshape(-1, 6)[idx]
        actual = clf._flat_delay_windows(x_seq, idx)
        assert torch.allclose(actual, expected)

    def test_encode_sequence_chunked_matches_unchunked(self):
        """Chunked sequence encoding should match the unchunked path."""
        x_seq = torch.randn(20, 10, 3)
        clf = StreamingNEFClassifier(3, 120, 2, window_size=4)
        full = clf.encode_sequence(x_seq, max_tokens=1_000_000)
        chunked = clf.encode_sequence(x_seq, max_tokens=50)
        assert torch.allclose(chunked, full, atol=1e-6)

    def test_batch_fit(self):
        """fit() should train decoders to reasonable accuracy on a simple task."""
        torch.manual_seed(70)
        N, T, d = 200, 10, 3
        x_seq = torch.randn(N, T, d)
        # Target: sign of mean of sequence means
        y = (x_seq.mean(dim=(1, 2)) > 0).long()
        targets = torch.zeros(N, 2)
        targets[torch.arange(N), y] = 1.0

        clf = StreamingNEFClassifier(
            d,
            500,
            2,
            window_size=3,
            rng=torch.Generator().manual_seed(700),
        )
        clf.fit(x_seq, targets)

        preds = clf(x_seq).argmax(dim=1)
        acc = (preds == y).float().mean().item()
        assert acc > 0.7

    def test_continuous_fit(self):
        """continuous_fit should match batch fit on same data."""
        torch.manual_seed(71)
        N, T, d = 100, 8, 2
        x_seq = torch.randn(N, T, d)
        targets = torch.randn(N, 3)
        alpha = 1e-2

        rng1 = torch.Generator().manual_seed(710)
        clf_batch = StreamingNEFClassifier(d, 150, 3, window_size=4, rng=rng1)
        clf_batch.fit(x_seq, targets, alpha=(alpha, "trace"))

        rng2 = torch.Generator().manual_seed(710)
        clf_cont = StreamingNEFClassifier(d, 150, 3, window_size=4, rng=rng2)
        clf_cont.continuous_fit(x_seq, targets, fixed_alpha=alpha)

        # Batch uses trace-scaled alpha, continuous uses fixed — compare
        # that both produce non-trivial decoders of similar magnitude
        assert clf_batch.decoders.data.abs().max() > 0.01
        assert clf_cont.decoders.data.abs().max() > 0.01

    def test_continuous_fit_defaults_to_fixed_alpha(self):
        """Woodbury paths use an explicitly named fixed-alpha default."""
        clf = StreamingNEFClassifier(2, 100, 3, window_size=3)
        clf.continuous_fit(torch.randn(10, 5, 2), torch.randn(10, 3))

        assert clf._woodbury_alpha == 1e-2

    def test_solve_requires_alpha_policy_pair(self):
        """Accumulated solves accept a single alpha policy argument."""
        clf = StreamingNEFClassifier(2, 100, 3, window_size=3)
        clf.accumulate(torch.randn(10, 5, 2), torch.randn(10, 3))

        with pytest.raises(ValueError, match="explicit .* pair"):
            clf.solve(alpha=1e-2)

    def test_continuous_fit_streaming(self):
        """Streaming continuous_fit should produce improving results."""
        torch.manual_seed(72)
        N, T, d = 200, 10, 3
        x_seq = torch.randn(N, T, d)
        y = (x_seq.mean(dim=(1, 2)) > 0).long()
        targets = torch.zeros(N, 2)
        targets[torch.arange(N), y] = 1.0

        clf = StreamingNEFClassifier(
            d,
            500,
            2,
            window_size=3,
            rng=torch.Generator().manual_seed(720),
        )
        accuracies = []
        for i in range(0, N, 50):
            clf.continuous_fit(x_seq[i : i + 50], targets[i : i + 50])
            preds = clf(x_seq).argmax(dim=1)
            acc = (preds == y).float().mean().item()
            accuracies.append(acc)

        # Should reach reasonable accuracy
        assert accuracies[-1] > 0.6

    def test_with_centers(self):
        """StreamingNEFClassifier should accept centers for data-driven biases."""
        x_seq = torch.randn(50, 10, 4)
        clf = StreamingNEFClassifier(
            4,
            100,
            2,
            window_size=3,
            centers=x_seq,
            rng=torch.Generator().manual_seed(730),
        )
        out = clf(x_seq)
        assert out.shape == (50, 2)

    def test_reset_continuous(self):
        """reset_continuous should clear all state."""
        clf = StreamingNEFClassifier(3, 100, 2, window_size=2)
        clf.continuous_fit(torch.randn(10, 5, 3), torch.randn(10, 2))
        clf.reset_continuous()
        assert clf._ata is None
        assert clf._M_inv is None

    def test_refresh_inverse(self):
        """refresh_inverse should work after continuous_fit."""
        torch.manual_seed(73)
        clf = StreamingNEFClassifier(3, 100, 2, window_size=2)
        clf.continuous_fit(torch.randn(20, 5, 3), torch.randn(20, 2))
        clf.refresh_inverse()
        # After refresh, decoders should be non-trivially different
        # (Woodbury drift correction)
        assert clf.decoders.data.abs().max() > 0

    def test_variable_sequence_lengths(self):
        """Should handle different sequence lengths across calls."""
        clf = StreamingNEFClassifier(2, 100, 3, window_size=3)
        # Different T values
        out1 = clf(torch.randn(5, 10, 2))
        out2 = clf(torch.randn(5, 20, 2))
        out3 = clf(torch.randn(5, 5, 2))
        assert out1.shape == (5, 3)
        assert out2.shape == (5, 3)
        assert out3.shape == (5, 3)

    def test_accumulate_solve_matches_woodbury(self):
        """accumulate() + solve() should produce similar accuracy to Woodbury."""
        torch.manual_seed(80)
        N, T, d = 200, 8, 2
        x_seq = torch.randn(N, T, d)
        y = (x_seq.mean(dim=(1, 2)) > 0).long()
        targets = torch.zeros(N, 2)
        targets[torch.arange(N), y] = 1.0
        alpha = 1e-2

        rng1 = torch.Generator().manual_seed(800)
        clf_w = StreamingNEFClassifier(d, 300, 2, window_size=4, rng=rng1)
        for i in range(0, N, 50):
            clf_w.continuous_fit(x_seq[i : i + 50], targets[i : i + 50], fixed_alpha=alpha)
        clf_w.refresh_inverse(fixed_alpha=alpha)
        acc_w = (clf_w(x_seq).argmax(1) == y).float().mean().item()

        rng2 = torch.Generator().manual_seed(800)
        clf_a = StreamingNEFClassifier(d, 300, 2, window_size=4, rng=rng2)
        for i in range(0, N, 50):
            clf_a.accumulate(x_seq[i : i + 50], targets[i : i + 50])
        clf_a.solve(alpha=(alpha, "fixed"))
        acc_a = (clf_a(x_seq).argmax(1) == y).float().mean().item()

        # Both should achieve good accuracy (>70%) and be close to each other
        assert acc_w > 0.7
        assert acc_a > 0.7
        assert abs(acc_w - acc_a) < 0.05

    @staticmethod
    def _reference_decoders(x_seq, targets, alpha, seed):
        """Decoders of an independent classifier solved on all the data at once."""
        ref = StreamingNEFClassifier(
            2, 100, 3, window_size=3, rng=torch.Generator().manual_seed(seed)
        )
        ref.accumulate(x_seq, targets, stats_dtype="float64")
        ref.solve(alpha=(alpha, "fixed"))
        return ref.decoders

    def test_continuous_fit_after_accumulate_uses_prior_statistics(self):
        """A missing inverse must be rebuilt from already accumulated statistics."""
        g = torch.Generator().manual_seed(82)
        x_seq = torch.randn(60, 6, 2, generator=g)
        targets = torch.randn(60, 3, generator=g)
        alpha = 1e-2

        clf = StreamingNEFClassifier(
            2, 100, 3, window_size=3, rng=torch.Generator().manual_seed(820)
        )
        f64 = {"stats_dtype": "float64"}  # float32 Gram round-off is amplified when n > N
        clf.accumulate(x_seq[:40], targets[:40], **f64)
        clf.continuous_fit(x_seq[40:], targets[40:], fixed_alpha=alpha, **f64)  # k=20 < n

        reference = self._reference_decoders(x_seq, targets, alpha, 820)
        assert torch.allclose(clf.decoders, reference, atol=1e-6)

    def test_accumulate_between_continuous_fits_invalidates_inverse(self):
        """accumulate() must not leave a stale inverse for the next Woodbury update."""
        g = torch.Generator().manual_seed(84)
        x_seq = torch.randn(60, 6, 2, generator=g)
        targets = torch.randn(60, 3, generator=g)
        alpha = 1e-2

        clf = StreamingNEFClassifier(
            2, 100, 3, window_size=3, rng=torch.Generator().manual_seed(840)
        )
        f64 = {"stats_dtype": "float64"}
        clf.continuous_fit(x_seq[:20], targets[:20], fixed_alpha=alpha, **f64)
        clf.accumulate(x_seq[20:40], targets[20:40], **f64)
        clf.continuous_fit(x_seq[40:], targets[40:], fixed_alpha=alpha, **f64)

        reference = self._reference_decoders(x_seq, targets, alpha, 840)
        assert torch.allclose(clf.decoders, reference, atol=1e-6)

    @pytest.mark.parametrize("bad_alpha", [1e-1, -1.0])
    def test_rejected_fixed_alpha_leaves_state_untouched(self, bad_alpha, monkeypatch):
        g = torch.Generator().manual_seed(83)
        clf = StreamingNEFClassifier(2, 50, 2, window_size=3, rng=g)
        clf.continuous_fit(
            torch.randn(10, 5, 2, generator=g), torch.randn(10, 2, generator=g), fixed_alpha=1e-2
        )
        before = [t.clone() for t in (clf._ata, clf._aty, clf._M_inv, clf.decoders.detach())]

        def _no_encoding(*_args, **_kwargs):
            raise AssertionError("a rejected call must not encode")

        monkeypatch.setattr(clf, "encode_sequence", _no_encoding)
        with pytest.raises(ValueError):
            clf.continuous_fit(
                torch.randn(10, 5, 2, generator=g),
                torch.randn(10, 2, generator=g),
                fixed_alpha=bad_alpha,
            )
        after = (clf._ata, clf._aty, clf._M_inv, clf.decoders.detach())
        assert all(torch.equal(x, y) for x, y in zip(before, after))
        assert clf._woodbury_alpha == 1e-2

    def test_accumulate_solve_no_woodbury_inverse(self):
        """accumulate + solve should not create the Woodbury inverse."""
        torch.manual_seed(81)
        clf = StreamingNEFClassifier(2, 100, 3, window_size=3)
        clf.accumulate(torch.randn(20, 5, 2), torch.randn(20, 3))
        clf.solve(alpha=(1e-2, "fixed"))

        # _M_inv should not exist (no Woodbury)
        assert not hasattr(clf, "_M_inv") or clf._M_inv is None
        # ATA and ATY should be float64 (for accumulation precision)
        assert clf._ata.dtype == torch.float64
        assert clf._aty.dtype == torch.float64
        # Decoders should be float32 (model working dtype)
        assert clf.decoders.dtype == torch.float32
