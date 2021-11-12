from __future__ import annotations

from dataclasses import dataclass
from typing import List

import probability_updating as pu


@dataclass
class Message:
    id: int
    outcomes: List[pu.outcome.Outcome]

    def __hash__(self):
        return hash(self.id)
