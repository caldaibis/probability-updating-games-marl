from __future__ import annotations

import json
import math
from abc import abstractmethod, ABC
from typing import Dict, List

import numpy as np
from ray.rllib.utils import softmax

import probability_updating as pu


class Action(ABC):
    def __init__(self, action):
        self.action = action

    @abstractmethod
    def __getitem__(self, idx) -> float:
        pass

    @classmethod
    @abstractmethod
    def from_array(cls, _input: np.ndarray, outcomes: List[pu.Outcome], messages: List[pu.Message], is_numpy_array: bool) -> Action:
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
        if agent == pu.Agent.Cont:
            return ContAction
        if agent == pu.Agent.Host:
            return HostAction
        return KeyError("Did you input a wrong agent name?")


class ContAction(Action):
    action: Dict[pu.Message, Dict[pu.Outcome, float]]

    def __getitem__(self, idx) -> float:
        return self.action[idx[1]][idx[0]]

    @classmethod
    def from_array(cls, _input: np.ndarray, outcomes: List[pu.Outcome], messages: List[pu.Message], is_numpy_array: bool) -> Action:
        i = 0
        strategy = {y: {x: 0 for x in outcomes} for y in messages}
        
        for y in messages:
            if len(y.outcomes) == 0:
                continue
                
            if len(y.outcomes) == 1:
                strategy[y][y.outcomes[0]] = 1
                continue

            tup = [_input[j] for j in range(i, i + len(y.outcomes))]
            if is_numpy_array:
                tup = softmax(np.array(tup))
            
            j = 0
            for x in y.outcomes:
                strategy[y][x] = tup[j]
                j += 1
                i += 1
                
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
    def from_array(cls, _input: np.ndarray, outcomes: List[pu.Outcome], messages: List[pu.Message], is_numpy_array: bool) -> Action:
        i = 0
        strategy = {x: {y: 0 for y in messages} for x in outcomes}
        
        for x in outcomes:
            if len(x.messages) == 0:
                continue
                
            if len(x.messages) == 1:
                strategy[x][x.messages[0]] = 1
                continue

            tup = [_input[j] for j in range(i, i + len(x.messages))]
            if is_numpy_array:
                tup = softmax(np.array(tup))
            
            j = 0
            for y in x.messages:
                strategy[x][y] = tup[j]
                j += 1
                i += 1
                
        return cls(strategy)

    def is_all_zero(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        return any(sum(self[x, y] for y in x.messages) == 0 for x in outcomes)

    def sums_to_one(self, outcomes: List[pu.Outcome], messages: List[pu.Message]) -> bool:
        return all(math.isclose(sum(self[x, y] for y in messages), 1, rel_tol=1e-5) for x in outcomes)
