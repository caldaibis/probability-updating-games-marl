from __future__ import annotations

from typing import List

import numpy as np

import src.probability_updating.games as games


class ExampleH(games.Game):
    @staticmethod
    def name() -> str:
        return "example_h"
    
    @staticmethod
    def pretty_name() -> str:
        return "Example H"

    @staticmethod
    def default_marginal() -> List[float]:
        return [
            1 / 5,
            3 / 5,
            1 / 5
        ]

    @staticmethod
    def message_structure() -> List[List[int]]:
        return [
            [0, 1],
            [1, 2],
            [0, 2]
        ]

    # x1 < y1, y3
    # x2 < y1, y2
    # x3 < y2, y3

    # y1 < x1, x2
    # y2 < x2, x3
    # y3 < x1, x3

    @staticmethod
    def cont_optimal_zero_one() -> np.ndarray:
        return np.array([[0., 1.], [1., 0.], [1 / 2, 1 / 2]])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([[.1, 0.], [1 / 2, 1 / 2], [1., 0.]])
