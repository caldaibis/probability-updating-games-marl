from __future__ import annotations

from typing import List

import numpy as np

import probability_updating as pu


def create_game(loss_cont: pu.LossFunc | pu.Loss, loss_quiz: pu.LossFunc | pu.Loss) -> pu.Game:
    outcomes, messages = pu.game.create_structure(marginal(), message_structure())
    marginal_outcome = dict(zip(outcomes, marginal()))
    return pu.Game(name(), outcomes, messages, marginal_outcome, loss_cont, loss_quiz)


def name() -> str:
    return "Monty Hall"


def marginal() -> List[float]:
    return [
        1 / 3,
        1 / 3,
        1 / 3
    ]


def message_structure() -> List[List[int]]:
    return [
        [0, 1],
        [1, 2]
    ]


def quiz_uniform() -> pu.PreStrategy:
    return pu.PreStrategy("quiz_uniform", np.array([1 / 2]))


def quiz_always_y1() -> pu.PreStrategy:
    return pu.PreStrategy("quiz_always_y1", np.array([1]))


def quiz_always_y2() -> pu.PreStrategy:
    return pu.PreStrategy("quiz_always_y2", np.array([0]))


def cont_always_switch() -> pu.PreStrategy:
    return pu.PreStrategy("cont_always_switch", np.array([1, 0]))


def cont_always_stay() -> pu.PreStrategy:
    return pu.PreStrategy("cont_always_stay", np.array([0, 1]))


def cont_min_loss_logarithmic() -> pu.PreStrategy:
    return pu.PreStrategy("cont_min_loss_logarithmic", np.array([2 / 3, 1 / 3]))

