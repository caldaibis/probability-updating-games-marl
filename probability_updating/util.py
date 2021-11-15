from __future__ import annotations
import math


def safe_log(p) -> float:
    if p == 0:
        return -math.inf
    return math.log(p)
