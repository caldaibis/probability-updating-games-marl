from __future__ import annotations
import math
from typing import List
import numpy as np


def safe_log(p) -> float:
    if p == 0:
        return -math.inf
    return math.log(p)


def sample_categorical_distribution(outcome_count: int) -> List[float]:
    """Samples a categorical/discrete distribution, uniform randomly."""
    return np.random.dirichlet([1] * outcome_count).tolist()

