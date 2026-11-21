"""leenef — machine learning with the Neural Engineering Framework."""

from .activations import ACTIVATIONS, make_activation
from .conv import (
    ConvNEFEnsemble,
    ConvNEFPipeline,
    ConvNEFStage,
    global_contrast_normalize,
    local_contrast_normalize,
)
from .encoders import ENCODER_STRATEGIES, make_encoders
from .ensemble import NEFEnsemble
from .layers import NEFLayer
from .networks import NEFNetwork
from .recurrent import RecurrentNEFLayer
from .rl import NEFAVRAgent, NEFAVREnsemble, NEFFeatures
from .rl_diag import (
    SparseChainEnv,
    SparseRewardDiagnostics,
    feature_coverage_stats,
    leverage_stats,
    solve_conditioning,
    sparse_metric_summary,
)
from .rl_sf import NEFHindsightSFAgent, NEFSFAgent, NEFSuccessor
from .selection import (
    agop,
    obs_saliency,
    phd_bases,
    prune_from_stats,
    psd_power,
    residual_moments,
    sample_directions_from_bases,
    sample_shaped_directions,
)
from .solvers import SOLVERS, solve_decoders, solve_from_normal_equations
from .streaming import StreamingNEFClassifier

__all__ = [
    "NEFLayer",
    "NEFEnsemble",
    "NEFNetwork",
    "RecurrentNEFLayer",
    "StreamingNEFClassifier",
    "ConvNEFStage",
    "ConvNEFPipeline",
    "ConvNEFEnsemble",
    "NEFFeatures",
    "NEFAVRAgent",
    "NEFAVREnsemble",
    "NEFHindsightSFAgent",
    "NEFSFAgent",
    "NEFSuccessor",
    "SparseChainEnv",
    "SparseRewardDiagnostics",
    "feature_coverage_stats",
    "leverage_stats",
    "solve_conditioning",
    "sparse_metric_summary",
    "local_contrast_normalize",
    "global_contrast_normalize",
    "make_encoders",
    "ENCODER_STRATEGIES",
    "make_activation",
    "ACTIVATIONS",
    "solve_decoders",
    "solve_from_normal_equations",
    "SOLVERS",
    "obs_saliency",
    "prune_from_stats",
    "residual_moments",
    "phd_bases",
    "sample_directions_from_bases",
    "agop",
    "psd_power",
    "sample_shaped_directions",
]
