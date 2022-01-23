from __future__ import annotations

import math
from typing import List

import numpy as np

import src.probability_updating as pu


def matrix_zero_one(outcome_count: int) -> np.ndarray:
    m = np.empty((outcome_count, outcome_count), dtype=int)
    for i in range(outcome_count):
        for j in range(outcome_count):
            if i == j:
                m[i, j] = 0
            else:
                m[i, j] = 1
    
    return m


def _zero_one_fn(cont: pu.ContAction, _: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    return 1 - cont[x, y]


def _brier_fn(cont: pu.ContAction, outcomes: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    loss: float = 0
    for x_ in outcomes:
        if x == x_:
            v = 1
        else:
            v = 0
        loss += math.pow(v - cont[x_, y], 2)
    
    return loss


def _logarithmic_fn(cont: pu.ContAction, _: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    return -pu.util.safe_log(cont[x, y])


def _matrix_fn(m: np.ndarray, cont: pu.ContAction, outcomes: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    return sum(cont[x, y] * m[x.id, x_prime.id] for x_prime in outcomes)


loss_fns = {
    pu.RANDOMISED_ZERO_ONE    : _zero_one_fn,
    pu.RANDOMISED_ZERO_ONE_NEG: lambda c, o, x, y: -_zero_one_fn(c, o, x, y),
    
    pu.BRIER                  : _brier_fn,
    pu.BRIER_NEG              : lambda c, o, x, y: -_brier_fn(c, o, x, y),
    
    pu.LOGARITHMIC            : _logarithmic_fn,
    pu.LOGARITHMIC_NEG        : lambda c, o, x, y: -_logarithmic_fn(c, o, x, y),
    
    pu.MATRIX                 : _matrix_fn,
    pu.MATRIX_NEG             : lambda m, c, o, x, y: -_matrix_fn(m, c, o, x, y),
}

loss_pretty_names = {
    pu.RANDOMISED_ZERO_ONE: 'randomised 0-1',
    pu.LOGARITHMIC: 'logarithmic',
    pu.BRIER: 'Brier'
}
