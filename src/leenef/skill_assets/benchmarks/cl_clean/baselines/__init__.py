"""Baselines registered into :mod:`benchmarks.cl_clean.experiments`.

Importing this subpackage is enough to register the MLP baselines as
extra cell values for the cross-method panel.
"""

from .mlp_baselines import er_run, ewc_run, finetune_run, joint_run

__all__ = ["finetune_run", "joint_run", "ewc_run", "er_run"]
