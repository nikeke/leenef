"""Clean-restart supervised learning benchmark infrastructure.

This package mirrors the discipline established in the RL clean sweeps
(now archived under ``attic/benchmarks``): a shared reproducibility
module, a resumable phase runner, and a sweep analyzer that decides
adoption against a base configuration via the rl_clean adoption rule.

Every experiment that contributes a numeric claim to Paper I writes its
results under ``results/sl_clean/`` with a code-identity hash, a per-seed
metric record, and enough metadata that re-running with the same code,
config, seed, and run context will skip the cell.

See ``docs/sl_clean_sweeps.md`` for the phase log and adoption history.
"""
