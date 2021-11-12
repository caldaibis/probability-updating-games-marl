from __future__ import annotations

from typing import List

import numpy as np

import probability_updating as pu


class MontyHall(pu.GameCreator):
    @staticmethod
    def name() -> str:
        return "montyHall"

    @staticmethod
    def marginal() -> List[float]:
        return [
            1 / 3,
            1 / 3,
            1 / 3
        ]

    @staticmethod
    def message_structure() -> List[List[int]]:
        return [
            [0, 1],
            [1, 2]
        ]

    @staticmethod
    def quiz_uniform() -> pu.StrategyWrapper:
        return pu.StrategyWrapper("quiz_uniform", np.array([1 / 2]))

    @staticmethod
    def quiz_always_y1() -> pu.StrategyWrapper:
        return pu.StrategyWrapper("quiz_always_y1", np.array([1]))

    @staticmethod
    def quiz_always_y2() -> pu.StrategyWrapper:
        return pu.StrategyWrapper("quiz_always_y2", np.array([0]))

    @staticmethod
    def cont_always_switch() -> pu.StrategyWrapper:
        return pu.StrategyWrapper("cont_always_switch", np.array([1, 0]))

    @staticmethod
    def cont_always_stay() -> pu.StrategyWrapper:
        return pu.StrategyWrapper("cont_always_stay", np.array([0, 1]))

    @staticmethod
    def cont_min_loss_logarithmic() -> pu.StrategyWrapper:
        return pu.StrategyWrapper("cont_min_loss_logarithmic", np.array([2 / 3, 1 / 3]))
