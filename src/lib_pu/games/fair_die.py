from __future__ import annotations

from typing import List

import numpy as np

import src.lib_pu.games as pu_games


class FairDie(pu_games.Game):
    @staticmethod
    def name() -> str:
        return pu_games.FAIR_DIE
    
    @staticmethod
    def pretty_name() -> str:
        return "Fair Die"

    @staticmethod
    def default_outcome_dist() -> List[float]:
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

    # x1 < y1
    # x2 < y1
    # x3 < y1, y2
    # x4 < y1, y2
    # x5 < y2
    # x6 < y2

    # y1 < x1, x2, x3, x4
    # y2 < x3, x4, x5, x5

    @staticmethod
    def cont_optimal_zero_one() -> np.ndarray:
        return np.array([[1 / 2, 1 / 2, 0., 0.], [0., 0., 1 / 2, 1 / 2]])

    @staticmethod
    def cont_optimal_matrix_ones_neg() -> np.ndarray:
        return np.array([[0., 0., 1., 0.], [1., 0., 0., 0.]])
    
    @staticmethod
    def cont_optimal_matrix_ones_neg2() -> np.ndarray:
        return np.array([[0., 0., 0.5, 0.5], [0.5, 0.5, 0., 0.]])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]])
