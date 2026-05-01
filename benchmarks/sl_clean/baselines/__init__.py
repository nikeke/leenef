"""Baselines for the SL clean comparison: MLP, LSTM, ESN, ELM.

Each baseline exposes a ``train_<name>(...)`` function that takes an
explicit ``torch.Generator`` for every stochastic operation (weight
init, dataloader shuffling, dropout if any). No reliance on the global
torch / numpy / random RNG inside the training loop. The cell-level
seed is consumed by ``benchmarks.sl_clean.experiments`` to derive
named sub-generators which are passed in here.
"""
