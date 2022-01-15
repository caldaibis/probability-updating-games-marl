from __future__ import annotations

import math
from typing import List, Optional
import probability_updating as pu
import probability_updating.util as util


def get_entropy_fn(loss: pu.Loss) -> Optional[pu.EntropyFunc]:
    return entropy_fns.get(loss.name)


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
    "randomised 0-1": _zero_one_fn,
    "brier": _brier_fn,
    "logarithmic": _logarithmic_fn,
}
