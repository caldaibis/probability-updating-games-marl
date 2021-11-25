from __future__ import annotations

from typing import Dict, List, Callable

import util
import probability_updating_env

from .message import Message
from .outcome import Outcome

from .agent import *
from .strategy import Strategy
from .simulation_wrapper import SimulationWrapper

from .loss import *
from .entropy import get_entropy_fn, entropy_fns

YgivenX = Dict[Outcome, Dict[Message, float]]
XgivenY = Dict[Message, Dict[Outcome, float]]

LossFunc = Callable[[XgivenY, List[Outcome], Outcome, Message], float]
EntropyFunc = Callable[[XgivenY, List[Outcome], Message], float]

inf_loss = 100
