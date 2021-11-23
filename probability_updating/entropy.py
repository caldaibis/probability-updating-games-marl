from __future__ import annotations

import math
from typing import List, Optional
import probability_updating as pu
import probability_updating.util as util


def get_entropy_fn(loss: pu.Loss) -> Optional[pu.EntropyFunc]:
    return entropy_fns.get(loss.name)


def _zero_one_fn(quiz_reverse: pu.XgivenY, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _max: float = 0
    for x in outcomes:
        if quiz_reverse[y][x] > _max:
            _max = quiz_reverse[y][x]

    return 1 - _max


def _brier_fn(quiz_reverse: pu.XgivenY, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _sum: float = 0
    for x in outcomes:
        _sum += math.pow(quiz_reverse[y][x], 2)

    return 1 - _sum


def _logarithmic_fn(quiz_reverse: pu.XgivenY, outcomes: List[pu.Outcome], y: pu.Message) -> float:
    _sum: float = 0
    for x in outcomes:
        e: float = -quiz_reverse[y][x] * util.safe_log(quiz_reverse[y][x])
        if not math.isnan(e):
            _sum += e

    return _sum


entropy_fns = {
    "randomised 0-1": _zero_one_fn,
    "brier": _brier_fn,
    "logarithmic": _logarithmic_fn,
}
