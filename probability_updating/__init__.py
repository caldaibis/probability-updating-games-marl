from __future__ import annotations

from typing import Dict, List, Callable

import util
import probability_updating_env

from .message import Message
from .outcome import Outcome

from .agent import *
from .strategy import Strategy
from .strategy_wrapper import StrategyWrapper
from .simulation_wrapper import SimulationWrapper

from .loss import *

YgivenX = Dict[Outcome, Dict[Message, float]]
XgivenY = Dict[Message, Dict[Outcome, float]]

LossFunc = Callable[[XgivenY, List[Outcome], Outcome, Message], float]
EntropyFunc = Callable[[XgivenY, List[Outcome], Message], float]

inf_loss = 1000
invalid_action_loss = inf_loss * 100
