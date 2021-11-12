from __future__ import annotations

from typing import List

import numpy as np

import probability_updating as pu
import probability_updating.games as games


class FairDie(games.Game):
    # Fair Die
    # y1 < {x1, x2, x3, x4}
    # y2 < {x3, x4, x5, x6}
    # --
    # x1 < {y1}
    # x2 < {y1}
    # x3 < {y1, y2}
    # x4 < {y1, y2}
    # x5 < {y2}
    # x6 < {y2}

    @staticmethod
    def name() -> str:
        return "fairDie"

    @staticmethod
    def marginal() -> List[float]:
        return [
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6
        ]

    @staticmethod
    def message_structure() -> List[List[int]]:
        return [
            [0, 1, 2, 3],
            [2, 3, 4, 5]
        ]

    @staticmethod
    def cont_always_switch() -> pu.StrategyWrapper:
        return pu.StrategyWrapper("cont_always_switch", np.array([1 / 2, 1 / 2, 0, 0, 0, 1 / 2]))

    @staticmethod
    def quiz_uniform() -> pu.StrategyWrapper:
        return pu.StrategyWrapper("quiz_uniform", np.array([1 / 2, 1 / 2]))
