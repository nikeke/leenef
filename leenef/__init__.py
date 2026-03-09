"""leenef — NEF supervised learning experiments."""

from .activations import ACTIVATIONS, make_activation
from .encoders import ENCODER_STRATEGIES, make_encoders
from .layers import NEFLayer
from .networks import NEFNetwork
from .recurrent import RecurrentNEFLayer
from .solvers import SOLVERS, solve_decoders, solve_from_normal_equations

__all__ = [
    "NEFLayer",
    "NEFNetwork",
    "RecurrentNEFLayer",
    "make_encoders",
    "ENCODER_STRATEGIES",
    "make_activation",
    "ACTIVATIONS",
    "solve_decoders",
    "solve_from_normal_equations",
    "SOLVERS",
]
