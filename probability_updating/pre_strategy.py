from __future__ import annotations

import numpy as np


class PreStrategy:
    name: str
    strategy: np.ndarray

    def __init__(self, name: str, strategy: np.ndarray):
        self.name = name
        self.strategy = strategy

    def __str__(self) -> str:
        return self.strategy.__str__()
