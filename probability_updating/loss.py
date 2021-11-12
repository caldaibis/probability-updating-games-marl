from __future__ import annotations

import math
from enum import Enum, auto
from typing import List

import numpy as np

import probability_updating as pu


def randomised_zero_one(cont: pu.XgivenY, _: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    return 1 - cont[y][x]


def brier(cont: pu.XgivenY, outcomes: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    loss: float = 0
    for x_ in outcomes:
        v = 1 if x == x_ else 0
        loss += math.pow(v - cont[y][x_], 2)

    return loss


def logarithmic(cont: pu.XgivenY, _: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    return -pu.util.safe_log(cont[y][x])


def hard_matrix_loss(m: np.ndarray, cont: pu.XgivenY, outcomes: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    for x_prime in outcomes:
        if cont[y][x_prime] == 1:
            return m[x.id, x_prime.id]

    return math.inf


def randomised_matrix_loss(m: np.ndarray, cont: pu.XgivenY, outcomes: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    return sum(cont[y][x] * m[x.id, x_prime.id] for x_prime in outcomes)


def matrix_zero_one(outcome_count: int) -> np.ndarray:
    m = np.empty((outcome_count, outcome_count), dtype=int)
    for i in range(outcome_count):
        for j in range(outcome_count):
            if i == j:
                m[i, j] = 0
            else:
                m[i, j] = 1

    return m


def randomised_zero_one_entropy(quiz_reverse: pu.XgivenY, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _max: float = 0
    for x in outcomes:
        if quiz_reverse[y][x] > _max:
            _max = quiz_reverse[y][x]

    return 1 - _max


def brier_entropy(quiz_reverse: pu.XgivenY, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _sum: float = 0
    for x in outcomes:
        _sum += math.pow(quiz_reverse[y][x], 2)

    return 1 - _sum


def logarithmic_entropy(quiz_reverse: pu.XgivenY, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _sum: float = 0
    for x in outcomes:
        e: float = -quiz_reverse[y][x] * pu.util.safe_log(quiz_reverse[y][x])
        if not math.isnan(e):
            _sum += e

    return _sum


def standard_loss(loss: Loss) -> pu.LossFunc:
    return {
        Loss.randomised_zero_one: randomised_zero_one,
        Loss.brier: brier,
        Loss.logarithmic: logarithmic,
    }[loss]


def standard_entropy(loss: Loss) -> pu.EntropyFunc:
    return {
        Loss.randomised_zero_one: randomised_zero_one_entropy,
        Loss.brier: brier_entropy,
        Loss.logarithmic: logarithmic_entropy
    }[loss]


class Loss(Enum):
    randomised_zero_one = auto(),
    brier = auto(),
    logarithmic = auto(),
    custom = auto()
