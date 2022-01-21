from __future__ import annotations

from typing import List

import numpy as np

import probability_updating.games as games


class MontyHall(games.Game):
    @staticmethod
    def name() -> str:
        return "monty_hall"

    @staticmethod
    def default_marginal() -> List[float]:
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
    def cont_optimal_zero_one() -> np.ndarray:
        return np.array([1, 0, 0, 1])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([1 / 2, 1 / 2])

    @staticmethod
    def cont_always_stay() -> np.ndarray:
        return np.array([0, 1])

    @staticmethod
    def cont_min_loss_logarithmic() -> np.ndarray:
        return np.array([2 / 3, 1 / 3])

    @staticmethod
    def host_always_y1() -> np.ndarray:
        return np.array([1])

    @staticmethod
    def host_always_y2() -> np.ndarray:
        return np.array([0])
