from __future__ import annotations

from typing import List

import numpy as np

import probability_updating.games as games


class MontyHall(games.Game):
    # Monty Hall
    # y1 < {x1, x2}
    # y2 < {x2, x3}
    # --
    # x1 < {y1}
    # x2 < {y1, y2}
    # x3 < {y2}

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
    def cont_always_switch() -> np.ndarray:
        return np.array([1, 0])

    @staticmethod
    def cont_always_stay() -> np.ndarray:
        return np.array([0, 1])

    @staticmethod
    def cont_min_loss_logarithmic() -> np.ndarray:
        return np.array([2 / 3, 1 / 3])

    @staticmethod
    def quiz_uniform() -> np.ndarray:
        return np.array([1 / 2])

    @staticmethod
    def quiz_always_y1() -> np.ndarray:
        return np.array([1])

    @staticmethod
    def quiz_always_y2() -> np.ndarray:
        return np.array([0])
