from __future__ import annotations

from typing import Type, Dict

import probability_updating as pu
import probability_updating.games as games
from pettingzoo.test import parallel_api_test


def monty_hall_test(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss]):
    game = game_type(losses)
    env = pu.probability_updating_env.env(game=game)

    parallel_api_test(env, num_cycles=100)
