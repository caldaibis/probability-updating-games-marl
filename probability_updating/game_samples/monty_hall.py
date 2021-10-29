from __future__ import annotations

from typing import List

import probability_updating as pu


def create_game() -> pu.Game:
    outcomes, messages = pu.game.create_structure(marginal(), message_structure())
    marginal_outcome = dict(zip(outcomes, marginal()))
    return pu.Game(name(), outcomes, messages, marginal_outcome)


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
    return pu.PreStrategy("quiz_uniform", {
        0: {
            0: 1,
            1: 0
        },
        1: {
            0: 1 / 2,
            1: 1 / 2
        },
        2: {
            0: 0,
            1: 1
        }
    })


def quiz_always_y1() -> pu.PreStrategy:
    return pu.PreStrategy("quiz_always_y1", {
        0: {
            0: 1,
            1: 0
        },
        1: {
            0: 1,
            1: 0
        },
        2: {
            0: 0,
            1: 1
        }
    })


def quiz_always_y2() -> pu.PreStrategy:
    return pu.PreStrategy("quiz_always_y2", {
        0: {
            0: 1,
            1: 0
        },
        1: {
            0: 0,
            1: 1
        },
        2: {
            0: 0,
            1: 1
        }
    })


def cont_always_switch() -> pu.PreStrategy:
    return pu.PreStrategy("cont_always_switch", {
        0: {
            0: 1,
            1: 0,
            2: 0
        },
        1: {
            0: 0,
            1: 0,
            2: 1,
        }
    })


def cont_always_stay() -> pu.PreStrategy:
    return pu.PreStrategy("cont_always_stay", {
        0: {
            0: 0,
            1: 1,
            2: 0
        },
        1: {
            0: 0,
            1: 1,
            2: 0,
        }
    })


def cont_min_loss_logarithmic() -> pu.PreStrategy:
    return pu.PreStrategy("cont_min_loss_logarithmic", {
        0: {
            0: 2 / 3,
            1: 1 / 3,
            2: 0
        },
        1: {
            0: 0,
            1: 1 / 3,
            2: 2 / 3,
        }
    })
