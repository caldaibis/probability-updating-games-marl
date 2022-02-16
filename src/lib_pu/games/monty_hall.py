from __future__ import annotations

from typing import List

import numpy as np

import src.lib_pu.games as pu_games


class MontyHall(pu_games.Game):
    @staticmethod
    def name() -> str:
        return pu_games.MONTY_HALL

    @staticmethod
    def default_outcome_dist() -> List[float]:
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

    # x1 < y1
    # x2 < y1, y2
    # x3 < y2

    # y1 < x1, x2
    # y2 < x2, x3

    @staticmethod
    def cont_default() -> np.ndarray:
        return np.array([[1., 0.], [0., 1.]])

    @staticmethod
    def host_default() -> np.ndarray:
        return np.array([[1 / 2, 1 / 2]])

    @staticmethod
    def cont_always_stay() -> np.ndarray:
        return np.array([[0., 1.], [1., 0.]])
    
    @staticmethod
    def cont_x0_x1_x2() -> np.ndarray:
        return np.array([[1., 0.], [1., 0.]])
    
    @staticmethod
    def cont_x2_x1_x0() -> np.ndarray:
        return np.array([[0., 1.], [0., 1.]])

    @staticmethod
    def cont_min_loss_logarithmic() -> np.ndarray:
        return np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])

    @staticmethod
    def host_always_y1() -> np.ndarray:
        return np.array([[1., 0.]])

    @staticmethod
    def host_always_y2() -> np.ndarray:
        return np.array([[0., 1.]])
