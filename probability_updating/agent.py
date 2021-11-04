from __future__ import annotations

from typing import List

Agent = str


def quiz() -> str:
    return "quiz"


def cont() -> str:
    return "cont"


def agents() -> List[str]:
    return [cont(), quiz()]
