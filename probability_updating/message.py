from __future__ import annotations

from typing import List

import probability_updating as pu


class Message:
    id: int
    outcomes: List[pu.outcome.Outcome]

    def __init__(self, _id: int):
        self.id = _id
        self.outcomes = []
