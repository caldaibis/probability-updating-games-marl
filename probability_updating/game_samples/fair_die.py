from __future__ import annotations

from typing import List

import probability_updating as pu


def create_game(loss_cont: pu.LossFunc | pu.Loss, loss_quiz: pu.LossFunc | pu.Loss) -> pu.Game:
    outcomes, messages = pu.game.create_structure(marginal(), message_structure())
    marginal_outcome = dict(zip(outcomes, marginal()))
    return pu.Game(name(), outcomes, messages, marginal_outcome, loss_cont, loss_quiz)


@property
def name() -> str:
    return "Fair Die"


@property
def marginal() -> List[float]:
    return [
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6
    ]


@property
def message_structure() -> List[List[int]]:
    return [
        [0, 1, 2, 3],
        [2, 3, 4, 5]
    ]
