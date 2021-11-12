from __future__ import annotations

from typing import List

import numpy as np

import probability_updating as pu


class FairDie(pu.GameCreator):
    @staticmethod
    def name() -> str:
        return "fairDie"

    @staticmethod
    def marginal() -> List[float]:
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
