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
