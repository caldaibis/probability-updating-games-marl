from __future__ import annotations

import math
from abc import abstractmethod, ABC
from typing import Dict, List

import numpy as np

import src.probability_updating as pu


class Action(ABC):
    def __init__(self, action):
        self.action = action

    @abstractmethod
    def __getitem__(self, idx) -> float:
        pass

    @classmethod
    @abstractmethod
    def from_array(cls, _input: np.ndarray, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> Action:
        pass

    @abstractmethod
    def is_all_zero(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        pass

    @abstractmethod
    def sums_to_one(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        pass

    def is_within_domain(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        return all(0 <= self[x, y] <= 1 for y in messages for x in outcomes)

    def __repr__(self) -> str:
        return repr(self.action)

    @staticmethod
    def get_class(agent: pu.Agent):
        if agent == pu.CONT:
            return ContAction
        if agent == pu.HOST:
            return HostAction
        return KeyError("Did you input a wrong agent name?")


class ContAction(Action):
    action: Dict[pu.Message, Dict[pu.Outcome, float]]

    def __getitem__(self, idx) -> float:
        return self.action[idx[1]][idx[0]]

    @classmethod
    def from_array(cls, _input: np.ndarray, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> Action:
        strategy = {y: {x: 0.0 for x in outcomes} for y in messages}

        i = 0
        cat = 0
        for y in messages:
            for x in y.outcomes:
                if len(y.outcomes) == 1:
                    strategy[y][x] = 1.0
                else:
                    strategy[y][x] = _input[cat][i]
                    i += 1
            if i > 0:
                cat += 1
                i = 0

        return cls(strategy)

    def is_all_zero(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        return any(sum(self[x, y] for x in y.outcomes) == 0 for y in messages)

    def sums_to_one(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        return all(math.isclose(sum(self[x, y] for x in outcomes), 1, rel_tol=1e-5) for y in messages)


class HostAction(Action):
    action: Dict[pu.Outcome, Dict[pu.Message, float]]

    def __getitem__(self, idx) -> float:
        return self.action[idx[0]][idx[1]]

    @classmethod
    def from_array(cls, _input: np.ndarray, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> Action:
        strategy = {x: {y: 0.0 for y in messages} for x in outcomes}

        i = 0
        cat = 0
        for x in outcomes:
            for y in x.messages:
                if len(x.messages) == 1:
                    strategy[x][y] = 1.0
                else:
                    strategy[x][y] = _input[cat][i]
                    i += 1
            if i > 0:
                cat += 1
                i = 0

        return cls(strategy)

    def is_all_zero(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        return any(sum(self[x, y] for y in x.messages) == 0 for x in outcomes)

    def sums_to_one(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        return all(math.isclose(sum(self[x, y] for y in messages), 1, rel_tol=1e-5) for x in outcomes)
