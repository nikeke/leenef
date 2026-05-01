"""H8 sanity: linear-activation NEFLayer = ordinary linear regression."""

import torch

from benchmarks.sl_clean.datasets import load_mnist
from leenef.layers import NEFLayer

bundle = load_mnist()
x_train, y_train_idx = bundle.x_train, bundle.y_train
x_test, y_test_idx = bundle.x_test, bundle.y_test
y_train = torch.zeros(len(y_train_idx), 10)
y_train[torch.arange(len(y_train_idx)), y_train_idx] = 1.0
print(f"MNIST: train={x_train.shape}, test={x_test.shape}")


def linreg(x, y, x_te, y_te, alpha=1e-3):
    n, d = x.shape
    XtX = x.T @ x
    XtX = XtX + alpha * (XtX.diagonal().mean()) * torch.eye(d) / d
    D = torch.linalg.solve(XtX, x.T @ y)
    pred = (x_te @ D).argmax(dim=1)
    return (pred == y_te).float().mean().item()


torch.manual_seed(0)
acc_lin = linreg(x_train, y_train, x_test, y_test_idx)
print("\n=== H8 Linear-collapse sanity check ===\n")
print(f"Linear ridge (784→10) baseline:                 {100 * acc_lin:.2f}%")

for n in [1000, 4000]:
    for biases_kw, label in [({"centers": x_train}, "data-driven"), ({}, "zero")]:
        torch.manual_seed(0)
        layer = NEFLayer(
            d_in=784,
            n_neurons=n,
            d_out=10,
            activation="linear",
            encoder_strategy="hypersphere",
            gain=(0.5, 2.0),
            **biases_kw,
        )
        layer.fit(x_train, y_train, alpha=1e-3)
        pred = layer(x_test).argmax(dim=1)
        acc = (pred == y_test_idx).float().mean().item()
        print(
            f"NEFLayer linear n={n:>4} ({label} biases): {100 * acc:.2f}%  (gap to linreg: {100 * (acc - acc_lin):+.2f} pp)"
        )

print("\nExpected (H8): all NEFLayer-linear rows within ~0.1 pp of linreg.")
print("This confirms §0 of Paper I: over-representation buys nothing without nonlinearity.\n")
