from __future__ import annotations

from typing import List

import numpy as np

import src.lib_pu.games as pu_games


class ExampleC(pu_games.Game):
    @staticmethod
    def name() -> str:
        return pu_games.EXAMPLE_C
    
    @staticmethod
    def pretty_name() -> str:
        return "Example C"

    @staticmethod
    def default_outcome_dist() -> List[float]:
        return [
            1 / 5,
            1 / 5,
            1 / 5,
            2 / 5
        ]

    @staticmethod
    def message_structure() -> List[List[int]]:
        return [
            [0, 1],
            [1, 2],
            [2, 3]
        ]

    # x1 < y1
    # x2 < y1, y2
    # x3 < y2, y3
    # x4 < y3

    # y1 < x1, x2
    # y2 < x2, x3
    # y3 < x3, x4

    @staticmethod
    def cont_optimal_zero_one() -> np.ndarray:
        return np.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2], [0., 1.]])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([[1., 0.], [0., 1.]])
