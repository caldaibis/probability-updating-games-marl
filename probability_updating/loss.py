from __future__ import annotations

import math
from typing import List

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
