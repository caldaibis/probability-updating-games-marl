from __future__ import annotations

from typing import List

import numpy as np

import src.lib_pu.games as pu_games


class ExampleH(pu_games.Game):
    @staticmethod
    def name() -> str:
        return pu_games.EXAMPLE_H
    
    @staticmethod
    def pretty_name() -> str:
        return pu_games.GAME_PRETTY_NAMES[pu_games.EXAMPLE_H]

    @staticmethod
    def default_outcome_dist() -> List[float]:
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
    def cont_default() -> np.ndarray:
        return np.array([[0., 1.], [1., 0.], [1 / 2, 1 / 2]])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([[.1, 0.], [1 / 2, 1 / 2], [1., 0.]])
