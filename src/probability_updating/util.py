from __future__ import annotations
import math
import random
from typing import List


def safe_log(p) -> float:
    if p == 0:
        return -math.inf
    return math.log(p)


def random_marginal_distribution(outcome_count: int) -> List[float]:
    """"Generates a marginal outcome distribution randomly.

    This version produces one probability uniformly, iteratively produces
    following distributions on the earlier ones. It seems skewed towards
    exponential-ish values, but it may be argued that it is less biased
    than the alternative version"""
    _max = 1
    distribution = []
    for i in range(outcome_count - 1):
        distribution.append(random.uniform(0, _max))
        _max -= distribution[i]

    distribution.append(_max)
    random.shuffle(distribution)

    assert(math.isclose(sum(distribution), 1, rel_tol=1e-5))

    return distribution


def random_marginal_distribution_alternative(outcome_count: int) -> List[float]:
    """"Generates a marginal outcome distribution randomly.

    This alternative version produces normalised distributions; the uniformly
    produced probabilities are divided by the sum of all probabilities. It is
    therefore more biased towards more mean-ish values."""
    distribution = []
    for i in range(outcome_count):
        distribution.append(random.random())

    _sum = sum(distribution)
    for i in range(outcome_count):
        distribution[i] /= _sum

    assert (math.isclose(sum(distribution), 1, rel_tol=1e-5))

    return distribution
