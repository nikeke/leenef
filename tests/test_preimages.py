import torch

from leenef.preimages import (
    activity_sign_preimage,
    nullspace_basis,
    onehot_preimages_from_anchors,
    sign_cell,
)


def test_activity_sign_preimage_is_exact_for_full_row_rank_encoder():
    W = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = torch.tensor([0.25, -0.5])
    target = torch.tensor([1.5, 2.0])
    signs = torch.tensor([-1.0, 1.0])

    x, diagnostics = activity_sign_preimage(W, b, target, signs)

    assert torch.allclose((W @ x + b).abs(), target, atol=1e-6)
    assert diagnostics["linear_residual_norm"] < 1e-6
    assert diagnostics["relative_activity_error"] < 1e-6


def test_onehot_preimages_report_in_cell_exactness():
    W = torch.eye(2)
    b = torch.zeros(2)
    decoders = torch.eye(2)
    anchors = torch.eye(2)

    prototypes, diagnostics = onehot_preimages_from_anchors(W, b, decoders, anchors)

    assert torch.allclose(prototypes, anchors, atol=1e-6)
    assert diagnostics["in_cell_count"] == 2.0
    assert diagnostics["max_linear_residual"] < 1e-6
    assert diagnostics["max_decode_dev"] < 1e-6


def test_nullspace_basis_spans_right_nullspace():
    matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    basis = nullspace_basis(matrix)

    assert basis.shape == (1, 3)
    assert torch.allclose(matrix @ basis.T, torch.zeros(2, 1), atol=1e-6)


def test_sign_cell_maps_zeros_to_positive_sign():
    W = torch.eye(2)
    b = torch.zeros(2)
    x = torch.tensor([0.0, -1.0])

    assert torch.equal(sign_cell(W, b, x), torch.tensor([1.0, -1.0]))
