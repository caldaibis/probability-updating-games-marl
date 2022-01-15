from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Message:
    id: int
    outcomes: List[probability_updating.outcome.Outcome]

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"y{str(self.id)}"

    def __repr__(self):
        return f"y{str(self.id)}"

    def __eq__(self, other: Message) -> bool:
        return self.id == other.id
