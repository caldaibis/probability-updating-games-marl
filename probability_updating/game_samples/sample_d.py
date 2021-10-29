from __future__ import annotations

from typing import List

import probability_updating as pu


def create_game() -> pu.Game:
    outcomes, messages = pu.game.create_structure(marginal(), message_structure())
    marginal_outcome = dict(zip(outcomes, marginal()))
    return pu.Game(name(), outcomes, messages, marginal_outcome)


@property
def name() -> str:
    return "Sample D"


@property
def marginal() -> List[float]:
    return [
        1 / 3,
        1 / 3,
        1 / 6,
        1 / 6
    ]


@property
def message_structure() -> List[List[int]]:
    return [
        [0, 1],
        [1, 2, 3]
    ]
