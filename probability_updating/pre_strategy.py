from __future__ import annotations

from typing import Dict


class PreStrategy:
    name: str
    strategy: Dict[int, Dict[int, float]]

    def __init__(self, name: str, strategy: Dict[int, Dict[int, float]]):
        self.name = name
        self.strategy = strategy

    def __str__(self) -> str:
        return self.strategy.__str__()
