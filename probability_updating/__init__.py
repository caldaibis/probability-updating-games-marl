from __future__ import annotations

from typing import Dict, List, Callable

from . import probability_updating_env
from . import strategy
from . import loss
from . import message
from . import outcome
from . import pre_strategy
from . import game
from . import util
from . import agent
from . import simulation_wrapper

from .probability_updating_env import env, ProbabilityUpdatingEnv
from .game import Game
from .loss import Loss, randomised_zero_one, brier, logarithmic, hard_matrix_loss,\
    randomised_matrix_loss, randomised_zero_one_entropy, brier_entropy, logarithmic_entropy, matrix_zero_one
from .message import Message
from .outcome import Outcome
from .strategy import Strategy
from .pre_strategy import PreStrategy
from .agent import Agent, agents, quiz, cont
from .simulation_wrapper import SimulationWrapper

YgivenX = Dict[Outcome, Dict[Message, float]]
XgivenY = Dict[Message, Dict[Outcome, float]]

LossFunc = Callable[[XgivenY, List[Outcome], Outcome, Message], float]
EntropyFunc = Callable[[XgivenY, List[Outcome], Message], float]

inf_loss = 100000000000
