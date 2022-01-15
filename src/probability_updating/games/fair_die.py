from __future__ import annotations

from typing import List

import numpy as np

import probability_updating.games as games


class FairDie(games.Game):
    @staticmethod
    def name() -> str:
        return "fair_die"

    @staticmethod
    def default_marginal() -> List[float]:
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
    def cont_optimal_zero_one() -> np.ndarray:
        return np.array([1 / 2, 1 / 2, 0, 0, 0, 1 / 2])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([1 / 2, 1 / 2])
