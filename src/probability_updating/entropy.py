from __future__ import annotations

import math
from typing import List, Optional, Callable
import src.probability_updating as pu
import src.probability_updating.util as util


def get_entropy_fn(loss: str) -> Optional[Callable[[pu.ContAction, List[pu.Outcome], pu.Message], float]]:
    return entropy_fns.get(loss)


def _zero_one_fn(host_reverse: pu.ContAction, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _max: float = 0
    for x in outcomes:
        if host_reverse[x, y] > _max:
            _max = host_reverse[x, y]

    return 1 - _max


def _brier_fn(host_reverse: pu.ContAction, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _sum: float = 0
    for x in outcomes:
        _sum += math.pow(host_reverse[x, y], 2)

    return 1 - _sum


def _logarithmic_fn(host_reverse: pu.ContAction, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _sum: float = 0
    for x in outcomes:
        e: float = -host_reverse[x, y] * util.safe_log(host_reverse[x, y])
        if not math.isnan(e):
            _sum += e

    return _sum


entropy_fns = {
    pu.RANDOMISED_ZERO_ONE: _zero_one_fn,
    
    pu.BRIER: _brier_fn,
    
    pu.LOGARITHMIC: _logarithmic_fn,
}
