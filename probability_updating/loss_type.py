from __future__ import annotations

LossType = str


def randomised_zero_one() -> LossType:
    return "randomise_zero_one"


def brier() -> LossType:
    return "brier"


def logarithmic() -> LossType:
    return "logarithmic"
