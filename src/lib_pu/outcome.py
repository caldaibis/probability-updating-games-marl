from __future__ import annotations

from dataclasses import dataclass
from typing import List

import src.lib_pu as pu


@dataclass
class Outcome:
    id: int
    messages: List[pu.message.Message]

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"x{str(self.id + 1)}"

    def __repr__(self):
        return f"x{str(self.id + 1)}"

    def __eq__(self, other: Outcome) -> bool:
        return self.id == other.id

    def old_str(self):
        return f"x{str(self.id)}"

    def pretty(self):
        return f"x_{str(self.id + 1)}"
