"""Tests for NEFEnsemble."""

import pytest
import torch

from leenef.ensemble import NEFEnsemble


class TestNEFEnsemble:
    def test_shape_mean(self):
        ens = NEFEnsemble(d_in=10, n_neurons=50, d_out=3, n_members=5, combine="mean")
        x = torch.randn(32, 10)
        y = ens(x)
        assert y.shape == (32, 3)

    def test_shape_vote(self):
        ens = NEFEnsemble(d_in=10, n_neurons=50, d_out=3, n_members=5, combine="vote")
        x = torch.randn(32, 10)
        y = ens(x)
        assert y.shape == (32, 3)

    def test_fit_and_forward(self):
        """Ensemble should be able to fit and predict a simple task."""
        ens = NEFEnsemble(d_in=1, n_neurons=200, d_out=1, n_members=5, combine="mean")
        x = torch.linspace(-1, 1, 100).unsqueeze(1)
        y = x**2
        ens.fit(x, y)
        pred = ens(x)
        mse = (pred - y).pow(2).mean().item()
        assert mse < 0.05

    def test_ensemble_beats_single(self):
        """Ensemble should be at least as accurate as the average single member."""
        torch.manual_seed(0)
        x = torch.cat(
            [
                torch.randn(100, 2) + torch.tensor([2.0, 0.0]),
                torch.randn(100, 2) + torch.tensor([-2.0, 0.0]),
                torch.randn(100, 2) + torch.tensor([0.0, 2.0]),
            ]
        )
        labels = torch.cat([torch.zeros(100), torch.ones(100), 2 * torch.ones(100)]).long()
        targets = torch.zeros(300, 3)
        targets.scatter_(1, labels.unsqueeze(1), 1.0)

        ens = NEFEnsemble(
            d_in=2, n_neurons=200, d_out=3, n_members=10, base_seed=0, combine="mean"
        )
        ens.fit(x, targets)

        # Ensemble accuracy
        ens_pred = ens(x).argmax(dim=1)
        ens_acc = (ens_pred == labels).float().mean().item()

        # Average individual member accuracy
        member_accs = []
        for member in ens.members:
            pred = member(x).argmax(dim=1)
            member_accs.append((pred == labels).float().mean().item())
        avg_member_acc = sum(member_accs) / len(member_accs)

        assert ens_acc >= avg_member_acc - 0.01  # ensemble should be at least as good

    def test_vote_mode(self):
        """Vote mode should return integer-like counts that sum to n_members."""
        ens = NEFEnsemble(d_in=5, n_neurons=100, d_out=3, n_members=7, combine="vote")
        x = torch.randn(20, 5)
        y = torch.randn(20, 3)
        ens.fit(x, y)
        votes = ens(x)
        # Each row should sum to n_members (each member casts one vote)
        assert torch.allclose(votes.sum(dim=1), torch.full((20,), 7.0))

    def test_different_seeds_different_encoders(self):
        """Members should have different encoders due to different seeds."""
        ens = NEFEnsemble(d_in=10, n_neurons=50, d_out=3, n_members=3, base_seed=42)
        e0 = ens.members[0].encoders
        e1 = ens.members[1].encoders
        assert not torch.equal(e0, e1)

    def test_state_dict_round_trip(self):
        """Save/load should preserve ensemble predictions."""
        ens = NEFEnsemble(d_in=5, n_neurons=100, d_out=2, n_members=3, base_seed=0)
        x = torch.randn(50, 5)
        y = torch.randn(50, 2)
        ens.fit(x, y)
        out1 = ens(x)

        state = ens.state_dict()
        ens2 = NEFEnsemble(d_in=5, n_neurons=100, d_out=2, n_members=3, base_seed=99)
        ens2.load_state_dict(state)
        out2 = ens2(x)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_bad_combine(self):
        with pytest.raises(ValueError, match="combine"):
            NEFEnsemble(d_in=5, n_neurons=50, d_out=3, combine="invalid")

    def test_with_centers(self):
        """Ensemble should accept data-driven centers."""
        centers = torch.randn(200, 10)
        ens = NEFEnsemble(d_in=10, n_neurons=100, d_out=3, n_members=3, centers=centers)
        x = torch.randn(32, 10)
        y = torch.randn(32, 3)
        ens.fit(x, y)
        pred = ens(x)
        assert pred.shape == (32, 3)

    def test_with_receptive_field(self):
        """Ensemble should work with receptive_field encoder strategy."""
        ens = NEFEnsemble(
            d_in=784,
            n_neurons=100,
            d_out=10,
            n_members=3,
            encoder_strategy="receptive_field",
            encoder_kwargs={"image_shape": (28, 28)},
        )
        x = torch.randn(20, 784)
        y = torch.randn(20, 10)
        ens.fit(x, y)
        pred = ens(x)
        assert pred.shape == (20, 10)
