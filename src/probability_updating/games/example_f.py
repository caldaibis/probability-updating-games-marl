from __future__ import annotations

from typing import List

import numpy as np

import probability_updating.games as games

# Matroid en Graph

class ExampleF(games.Game):
    @staticmethod
    def name() -> str:
        return "example_f"

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
            [1, 2],
            [0, 2]
        ]

    @staticmethod
    def cont_optimal_zero_one() -> list:
        return [1 / 2, 1 / 2, 1 / 2]

    @staticmethod
    def host_default() -> list:
        return [1 / 2, 1 / 2, 1 / 2]
