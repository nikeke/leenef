"""leenef — NEF supervised learning experiments."""

from .activations import ACTIVATIONS, make_activation
from .conv import ConvNEFPipeline, ConvNEFStage
from .encoders import ENCODER_STRATEGIES, make_encoders
from .ensemble import NEFEnsemble
from .layers import NEFLayer
from .networks import NEFNetwork
from .recurrent import RecurrentNEFLayer
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
    "make_encoders",
    "ENCODER_STRATEGIES",
    "make_activation",
    "ACTIVATIONS",
    "solve_decoders",
    "solve_from_normal_equations",
    "SOLVERS",
]
