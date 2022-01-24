from __future__ import annotations

MONTY_HALL = "monty_hall"
FAIR_DIE = "fair_die"
EXAMPLE_C = "example_c"
EXAMPLE_D = "example_d"
EXAMPLE_E = "example_e"
EXAMPLE_F = "example_f"
EXAMPLE_H = "example_h"

CLIPPED_INFINITY_LOSS = 5

RANDOMISED_ZERO_ONE = "randomised_0_1"
RANDOMISED_ZERO_ONE_NEG = RANDOMISED_ZERO_ONE + "_neg"

BRIER = "brier"
BRIER_NEG = BRIER + "_neg"

LOGARITHMIC = "logarithmic"
LOGARITHMIC_NEG = LOGARITHMIC + "_neg"

MATRIX = "matrix"
MATRIX_NEG = MATRIX + "_neg"

from src.probability_updating.probability_updating_env import ProbabilityUpdatingEnv
from src.probability_updating.probability_updating_env_wrapper import ProbabilityUpdatingEnvWrapper
from src.probability_updating.games import Game

from src.probability_updating.action import Action, ContAction, HostAction
from src.probability_updating.message import Message
from src.probability_updating.outcome import Outcome

from src.probability_updating.agent import *
from src.probability_updating.strategy_util import StrategyUtil
from src.probability_updating.simulation_wrapper import SimulationWrapper

from src.probability_updating.loss import *
from src.probability_updating.entropy import get_entropy_fn, entropy_fns

from src.probability_updating.util import sample_categorical_distribution
