from __future__ import annotations

from typing import Callable

from .probability_updating_env import ProbabilityUpdatingEnv
from .games import Game

from .action import Action, ContAction, HostAction
from .message import Message
from .outcome import Outcome

from .agent import *
from .strategy_util import StrategyUtil
from .simulation_wrapper import SimulationWrapper

from .loss import *
from .entropy import get_entropy_fn, entropy_fns

from .util import random_marginal_distribution_alternative, random_marginal_distribution

LossFunc = Callable[[ContAction, List[Outcome], Outcome, Message], float]
EntropyFunc = Callable[[ContAction, List[Outcome], Message], float]

inf_loss = 5
