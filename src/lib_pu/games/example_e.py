from __future__ import annotations

from typing import List

import numpy as np

import src.lib_pu.games as pu_games


class ExampleE(pu_games.Game):
    @staticmethod
    def name() -> str:
        return pu_games.EXAMPLE_E

    @staticmethod
    def default_outcome_dist() -> List[float]:
        return [
            0.45,
            0.05,
            0.25,
            0.25
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
    def cont_default() -> np.ndarray:
        return np.array([[1., 0.], [0., 1 / 2, 1 / 2]])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([[1., 0.]])
    
    @staticmethod
    def cont_1() -> np.ndarray:
        return np.array([[1., 0.], [0., 0.4, 0.6]])

    @staticmethod
    def host_1() -> np.ndarray:
        return np.array([[0., 1.]])
