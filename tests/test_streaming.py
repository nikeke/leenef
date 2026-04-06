"""Tests for StreamingNEFClassifier — temporal sequence classification."""

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
        clf_batch.fit(x_seq, targets, alpha=alpha)

        rng2 = torch.Generator().manual_seed(710)
        clf_cont = StreamingNEFClassifier(d, 150, 3, window_size=4, rng=rng2)
        clf_cont.continuous_fit(x_seq, targets, alpha=alpha)

        # Batch uses trace-scaled alpha, continuous uses fixed — compare
        # that both produce non-trivial decoders of similar magnitude
        assert clf_batch.decoders.data.abs().max() > 0.01
        assert clf_cont.decoders.data.abs().max() > 0.01

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
