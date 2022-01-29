from __future__ import annotations

import math
from typing import List

import numpy as np

import src.pu_lib as pu


CLIPPED_INFINITY_LOSS = 5

RANDOMISED_ZERO_ONE = "randomised_0_1"
RANDOMISED_ZERO_ONE_NEG = RANDOMISED_ZERO_ONE + "_neg"

BRIER = "brier"
BRIER_NEG = BRIER + "_neg"

LOGARITHMIC = "logarithmic"
LOGARITHMIC_NEG = LOGARITHMIC + "_neg"

MATRIX = "matrix"

LOSS_LIST = [
    RANDOMISED_ZERO_ONE,
    BRIER,
    LOGARITHMIC,
    MATRIX,
]

LOSS_NEG_LIST = [
    RANDOMISED_ZERO_ONE_NEG,
    BRIER_NEG,
    LOGARITHMIC_NEG,
]

LOSS_PAIRS_COOPERATIVE = list(zip(LOSS_LIST, LOSS_LIST))
LOSS_PAIRS_ZERO_SUM = list(zip(LOSS_LIST, LOSS_NEG_LIST))
LOSS_PAIRS_ALL = [*LOSS_PAIRS_ZERO_SUM, *LOSS_PAIRS_COOPERATIVE]


def safe_log(x) -> float:
    return math.log(x) if x != 0 else -math.inf


def matrix_random_cooperative(outcome_count: int) -> np.ndarray:
    m = np.random.random((outcome_count, outcome_count))
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_random_competitive(outcome_count: int) -> np.ndarray:
    m = -1 * np.random.random((outcome_count, outcome_count))
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_random_mixed(outcome_count: int) -> np.ndarray:
    m = -2 * np.random.random((outcome_count, outcome_count)) + 1
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_ones(outcome_count: int) -> np.ndarray:
    m = np.ones((outcome_count, outcome_count))
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_ones_neg(outcome_count: int) -> np.ndarray:
    m = -1 * np.ones((outcome_count, outcome_count))
    for x in range(outcome_count):
        m[x, x] = 0
    
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
    return -safe_log(cont[x, y])


def _matrix_fn(m: np.ndarray, cont: pu.ContAction, outcomes: List[pu.Outcome], x: pu.Outcome, y: pu.Message) -> float:
    return sum(cont[x_prime, y] * m[x.id, x_prime.id] for x_prime in outcomes)


LOSS_FN_LIST = {
    RANDOMISED_ZERO_ONE    : _zero_one_fn,
    RANDOMISED_ZERO_ONE_NEG: lambda c, o, x, y: -_zero_one_fn(c, o, x, y),
    BRIER                  : _brier_fn,
    BRIER_NEG              : lambda c, o, x, y: -_brier_fn(c, o, x, y),
    LOGARITHMIC            : _logarithmic_fn,
    LOGARITHMIC_NEG        : lambda c, o, x, y: -_logarithmic_fn(c, o, x, y),
    MATRIX                 : _matrix_fn,
}

LOSS_PRETTY_NAME_LIST = {
    RANDOMISED_ZERO_ONE: 'randomised 0-1',
    BRIER: 'Brier',
    LOGARITHMIC: 'logarithmic',
    MATRIX: 'matrix'
}


def proper_entropy_fn(loss_fn, host_reverse: pu.ContAction, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _sum: float = 0
    for x in outcomes:
        e: float = host_reverse[x, y] * loss_fn(host_reverse, outcomes, x, y)
        if not math.isnan(e):
            _sum += e

    return _sum


def randomised_entropy_fn(_, host_reverse: pu.ContAction, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    return 1 - max(host_reverse[x, y] for x in outcomes)


ENTROPY_FN_LIST = {
    RANDOMISED_ZERO_ONE: lambda l, p, o, y:      randomised_entropy_fn(l, p, o, y),
    RANDOMISED_ZERO_ONE_NEG: lambda l, p, o, y: -randomised_entropy_fn(l, p, o, y),
    
    BRIER: proper_entropy_fn,
    BRIER_NEG: proper_entropy_fn,
    
    LOGARITHMIC: proper_entropy_fn,
    LOGARITHMIC_NEG: proper_entropy_fn,
    
    MATRIX: randomised_entropy_fn,
}
