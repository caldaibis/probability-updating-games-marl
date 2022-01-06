from __future__ import annotations

from typing import Dict

import numpy as np


class InvalidStrategyError(Exception):
    """Exception raised for errors in the strategy input.

        Attributes:
            strategy -- the given strategy
            shape -- shape of the correct strategy space
        """

    strategy: np.ndarray
    shape: int

    def __init__(self, strategy: np.ndarray, shape: int):
        self.strategy = strategy
        self.shape = shape

    def __str__(self):
        return f'Strategy: {self.strategy} is invalid. It might be swapped (cont/host) or not the correct shape ({self.shape})?'


class AllZeroesError(Exception):
    """Exception raised for errors in the strategy input regarding all zeroes.

        Attributes:
            strategy -- the given strategy
            shape -- shape of the correct strategy space
        """

    strategy: Dict[int, Dict[int, float]]
    shape: int

    def __init__(self, strategy: Dict[int, Dict[int, float]], shape: int):
        super().__init__(strategy, shape)

    def __str__(self):
        return f'Strategy: {self.strategy} is invalid because it has all zeroes for a particular outcome/message!'


class NotWithinDomainError(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class NoDistributionError(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message
