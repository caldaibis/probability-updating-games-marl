from __future__ import annotations
import math


def safe_log(x) -> float:
    if x == 0:
        return -math.inf
    return math.log(x)
