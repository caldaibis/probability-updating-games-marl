from __future__ import annotations

from typing import List

import numpy as np

import probability_updating as pu


class SampleD(pu.GameCreator):
    @staticmethod
    def name() -> str:
        return "sampleD"

    @staticmethod
    def marginal() -> List[float]:
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
