from __future__ import annotations

from typing import List

import probability_updating as pu


class Outcome:
    id: int
    messages: List[pu.message.Message]

    def __init__(self, _id: int):
        self.id = _id
        self.messages = []
