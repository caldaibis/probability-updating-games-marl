from __future__ import annotations

import math
from typing import List

import numpy as np

import src.lib_pu as pu


CLIPPED_INFINITY_LOSS = 5

RANDOMISED_ZERO_ONE = "randomised_0_1"
RANDOMISED_ZERO_ONE_NEG = RANDOMISED_ZERO_ONE + "_neg"

BRIER = "brier"
BRIER_NEG = BRIER + "_neg"

LOGARITHMIC = "logarithmic"
LOGARITHMIC_NEG = LOGARITHMIC + "_neg"

MATRIX = "matrix"
MATRIX_CUSTOM_3 = "matrix_custom_3"
MATRIX_CUSTOM_3_NEG = "matrix_custom_3_neg"

MATRIX_CUSTOM_6 = "matrix_custom_6"
MATRIX_CUSTOM_6_NEG = "matrix_custom_6_neg"

MATRIX_PREDEFINED_POS = [f"matrix_predefined_pos_{i}" for i in range(15)]
MATRIX_PREDEFINED_NEG = [f"matrix_predefined_neg_{i}" for i in range(15)]
MATRIX_PREDEFINED = [*MATRIX_PREDEFINED_POS, *MATRIX_PREDEFINED_NEG]

MATRIX_ONES_POS = 'matrix_ones_pos'
MATRIX_ONES_NEG = 'matrix_ones_neg'
MATRIX_RAND_POS = [f"matrix_rand_pos_{i}" for i in range(10)]
MATRIX_RAND_NEG = [f"matrix_rand_neg_{i}" for i in range(10)]
MATRIX_RAND_MIX = [f"matrix_rand_mix_{i}" for i in range(10)]
MATRIX_RAND = [*MATRIX_RAND_POS, *MATRIX_RAND_NEG, *MATRIX_RAND_MIX]

LOSSES = [
    RANDOMISED_ZERO_ONE,
    BRIER,
    LOGARITHMIC,
    MATRIX,
]

LOSSES_NEG = [
    RANDOMISED_ZERO_ONE_NEG,
    BRIER_NEG,
    LOGARITHMIC_NEG,
]

LOSS_PAIRS_COOPERATIVE = list(zip(LOSSES, LOSSES))
LOSS_PAIRS_ZERO_SUM = list(zip(LOSSES, LOSSES_NEG))
LOSS_ALL_PAIRS = [*LOSS_PAIRS_ZERO_SUM, *LOSS_PAIRS_COOPERATIVE]


def safe_log(x) -> float:
    return math.log(x) if x != 0 else -math.inf


def matrix_random_pos(outcome_count: int) -> np.ndarray:
    m = np.random.random((outcome_count, outcome_count))
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_random_neg(outcome_count: int) -> np.ndarray:
    m = -1 * np.random.random((outcome_count, outcome_count))
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_random_mix(outcome_count: int) -> np.ndarray:
    m = -2 * np.random.random((outcome_count, outcome_count)) + 1
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_ones_pos(outcome_count: int) -> np.ndarray:
    m = np.ones((outcome_count, outcome_count))
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_ones_neg(outcome_count: int) -> np.ndarray:
    m = -1 * np.ones((outcome_count, outcome_count))
    for x in range(outcome_count):
        m[x, x] = 0
    
    return m


def matrix_custom_3() -> np.ndarray:
    return np.array(
        [
            [0, 1, 500],
            [1, 0, 1],
            [500, 1, 0],
        ])


def matrix_custom_3_neg() -> np.ndarray:
    return -1 * matrix_custom_3()


def matrix_custom_6() -> np.ndarray:
    return np.array(
        [
            [0, 1, 1, 1, -500, -500],
            [1, 0, 1, 1, -500, -500],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1],
            [-500, -500, 1, 1, 0, 1],
            [-500, -500, 1, 1, 1, 0],
        ])

def matrix_custom_3() -> np.ndarray:
    return np.array(
        [
            [0, 1, 500],
            [1, 0, 1],
            [500, 1, 0],
        ])

def matrix_custom_6_neg() -> np.ndarray:
    return -1 * matrix_custom_6()


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


LOSS_FNS = {
    RANDOMISED_ZERO_ONE    : _zero_one_fn,
    RANDOMISED_ZERO_ONE_NEG: lambda c, o, x, y: -_zero_one_fn(c, o, x, y),
    BRIER                  : _brier_fn,
    BRIER_NEG              : lambda c, o, x, y: -_brier_fn(c, o, x, y),
    LOGARITHMIC            : _logarithmic_fn,
    LOGARITHMIC_NEG        : lambda c, o, x, y: -_logarithmic_fn(c, o, x, y),
    MATRIX                 : _matrix_fn,
}

LOSS_NAMES = {
    RANDOMISED_ZERO_ONE: 'randomised 0-1',
    BRIER: 'Brier',
    LOGARITHMIC: 'logarithmic',
    MATRIX: 'matrix'
}


def _proper_entropy_fn(loss_fn, host_reverse: pu.ContAction, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _sum: float = 0
    for x in outcomes:
        e: float = host_reverse[x, y] * loss_fn(host_reverse, outcomes, x, y)
        if not math.isnan(e):
            _sum += e

    return _sum


def _randomised_entropy_fn(_, host_reverse: pu.ContAction, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    return 1 - max(host_reverse[x, y] for x in outcomes)


def _matrix_entropy_fn(m: np.ndarray, _, host_reverse: pu.ContAction, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    minimal_sum = math.inf
    for x in outcomes:
        _sum = 0
        for x_prime in outcomes:
            _sum += host_reverse[x_prime, y] * m[x_prime.id, x.id]
        minimal_sum = min(minimal_sum, _sum)
    return minimal_sum
    

ENTROPY_FNS = {
    RANDOMISED_ZERO_ONE: _randomised_entropy_fn,
    RANDOMISED_ZERO_ONE_NEG: lambda l, p, o, y: -_randomised_entropy_fn(l, p, o, y),
    
    BRIER: _proper_entropy_fn,
    BRIER_NEG: _proper_entropy_fn,
    
    LOGARITHMIC: _proper_entropy_fn,
    LOGARITHMIC_NEG: _proper_entropy_fn,
    
    MATRIX: _matrix_entropy_fn,
}
