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
from .rl import NEFFeatures, NEFFQIAgent, NEFFQIEnsemble
from .rl_hindsight import (
    acrobot_end_effector_height,
    append_goal,
    goal_reached,
    hindsight_event_targets,
    mountaincar_achieved_goal,
    terminal_hindsight_goal,
)
from .rl_novelty import leverage_scores, nef_fqi_action_novelty, novelty_scores
from .rl_successor import NEFSuccessorFeatureAgent, truncated_successor_targets
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
    "NEFFQIAgent",
    "NEFFQIEnsemble",
    "NEFSuccessorFeatureAgent",
    "truncated_successor_targets",
    "leverage_scores",
    "novelty_scores",
    "nef_fqi_action_novelty",
    "append_goal",
    "mountaincar_achieved_goal",
    "acrobot_end_effector_height",
    "goal_reached",
    "hindsight_event_targets",
    "terminal_hindsight_goal",
    "local_contrast_normalize",
    "global_contrast_normalize",
    "make_encoders",
    "ENCODER_STRATEGIES",
    "make_activation",
    "ACTIVATIONS",
    "solve_decoders",
    "solve_from_normal_equations",
    "SOLVERS",
]
