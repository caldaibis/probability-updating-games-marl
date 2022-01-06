from __future__ import annotations

from dataclasses import dataclass
from typing import List

import probability_updating as pu


@dataclass
class Outcome:
    id: int
    messages: List[pu.message.Message]

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"x{str(self.id)}"

    def __repr__(self):
        return f"x{str(self.id)}"

    def __eq__(self, other: Outcome) -> bool:
        return self.id == other.id
