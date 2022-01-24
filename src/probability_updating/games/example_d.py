from __future__ import annotations

from typing import List

import numpy as np

import src.probability_updating.games as games


class ExampleD(games.Game):
    @staticmethod
    def name() -> str:
        return "example_d"
    
    @staticmethod
    def pretty_name() -> str:
        return "Example D"

    @staticmethod
    def default_marginal() -> List[float]:
        return [
            1 / 3,
            1 / 3,
            1 / 6,
            1 / 6
        ]

    @staticmethod
    def message_structure() -> List[List[int]]:
        return [
            [0, 1],
            [1, 2, 3]
        ]

    # x1 < y1
    # x2 < y1, y2
    # x3 < y3
    # x4 < y3

    # y1 < x1, x2
    # y2 < x2, x3, x4

    @staticmethod
    def cont_optimal_zero_one() -> np.ndarray:
        return np.array([[1., 0.], [1 / 2, 1 / 2, 0.]])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([[1 / 2, 1 / 2]])
