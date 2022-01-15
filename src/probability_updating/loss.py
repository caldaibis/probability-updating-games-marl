from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

import probability_updating as pu
import probability_updating.util as util


@dataclass
class Loss:
    name: str
    _loss_fn: pu.LossFunc

    def __call__(self, *args, **kwargs):
        return self._loss_fn(*args)

    @staticmethod
    def null() -> Loss:
        """An empty loss function, always returning zero. Used for zero-sum games with parameter sharing"""
        return Loss("null", Loss._null_fn)

    @staticmethod
    def zero_one() -> Loss:
        """Randomised 0-1 loss"""
        return Loss("randomised_0-1", Loss._zero_one_fn)

    @staticmethod
    def zero_one_negative() -> Loss:
        """Negative randomised 0-1 loss"""
        return Loss("negative_randomised 0-1", lambda c, o, x, y: -Loss._zero_one_fn(c, o, x, y))

    @staticmethod
    def brier() -> Loss:
        """Brier loss"""
        return Loss("brier", Loss._brier_fn)

    @staticmethod
    def brier_negative() -> Loss:
        """Negative Brier loss"""
        return Loss("negative_brier", lambda c, o, x, y: -Loss._brier_fn(c, o, x, y))

    @staticmethod
    def logarithmic() -> Loss:
        """Logarithmic loss"""
        return Loss("logarithmic", Loss._logarithmic_fn)

    @staticmethod
    def logarithmic_negative() -> Loss:
        """negative logarithmic loss"""
        return Loss("negative_logarithmic", lambda c, o, x, y: -Loss._logarithmic_fn(c, o, x, y))

    @staticmethod
    def matrix(m: np.ndarray) -> Loss:
        """Matrix loss"""
        return Loss("matrix", lambda c, o, x, y: Loss._matrix_fn(m, c, o, x, y))

    @staticmethod
    def matrix_negative(m: np.ndarray) -> Loss:
        """Negative matrix loss"""
        return Loss("negative_matrix", lambda c, o, x, y: -Loss._matrix_fn(m, c, o, x, y))

    @staticmethod
    def _null_fn(_, __, ___, ____) -> float:
        return 0

    @staticmethod
    def _zero_one_fn(cont: pu.ContAction, _: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
        return 1 - cont[x, y]

    @staticmethod
    def _brier_fn(cont: pu.ContAction, outcomes: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
        loss: float = 0
        for x_ in outcomes:
            if x == x_:
                v = 1
            else:
                v = 0
            loss += math.pow(v - cont[x_, y], 2)

        return loss

    @staticmethod
    def _logarithmic_fn(cont: pu.ContAction, _: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
        return -util.safe_log(cont[x, y])

    @staticmethod
    def _matrix_fn(m: np.ndarray, cont: pu.ContAction, outcomes: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
        return sum(cont[x, y] * m[x.id, x_prime.id] for x_prime in outcomes)

    @staticmethod
    def matrix_zero_one(outcome_count: int) -> np.ndarray:
        m = np.empty((outcome_count, outcome_count), dtype=int)
        for i in range(outcome_count):
            for j in range(outcome_count):
                if i == j:
                    m[i, j] = 0
                else:
                    m[i, j] = 1

        return m

    # @staticmethod
    # def hard_matrix_loss(m: np.ndarray, cont: pu.XgivenY, outcomes: List[pu.Outcome], x: pu.Outcome,
    #                      y: pu.Message) -> float:
    #     for x_prime in outcomes:
    #         if cont[y][x_prime] == 1:
    #             return m[x.id, x_prime.id]
    #
    #     return math.inf
