from __future__ import annotations

from typing import List

import numpy as np

import probability_updating.games as games

# Matroid en Graph


class ExampleH(games.Game):
    @staticmethod
    def name() -> str:
        return "example_h"

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

    @staticmethod
    def cont_optimal_zero_one() -> list:
        return [0, 1, 1 / 2]

    @staticmethod
    def host_default() -> list:
        return [1, 1 / 2, 1]
