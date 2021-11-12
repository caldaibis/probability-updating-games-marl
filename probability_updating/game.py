from __future__ import annotations

from types import FunctionType
import math
from abc import abstractmethod
from typing import List, Dict, Callable, Optional

import numpy as np

import probability_updating as pu


class Game:
    outcomes: List[pu.Outcome]
    messages: List[pu.Message]

    marginal_outcome: Dict[pu.Outcome, float]
    marginal_message: Dict[pu.Message, float]

    loss_fn: Dict[pu.Agent, Callable[[pu.XgivenY, pu.Outcome, pu.Message], float]] = {}
    entropy_fn: Dict[pu.Agent, Callable[[pu.Message], float]] = {}

    strategy: pu.Strategy
    _quiz: pu.YgivenX
    quiz_reverse: pu.XgivenY
    _cont: pu.XgivenY

    def __init__(self,
                 outcomes: List[pu.Outcome],
                 messages: List[pu.Message],
                 marginal_outcome: Dict[pu.Outcome, float],
                 loss_cont: pu.LossFunc | pu.Loss,
                 loss_quiz: pu.LossFunc | pu.Loss):
        self.strategy = pu.Strategy(self)

        self.outcomes = outcomes
        self.messages = messages
        self.marginal_outcome = marginal_outcome

        if isinstance(loss_cont, FunctionType):
            self.loss_fn[pu.cont()] = lambda cont, x, y: loss_cont(cont, self.outcomes, x, y)
            self.entropy_fn[pu.cont()] = NotImplemented
        elif isinstance(loss_cont, pu.Loss):
            # Wat de fuck, attribute nog niet gedefinieerd in lamda functie maar in werkelijkheid wel gedefinieerd, wat te doen?
            self.loss_fn[pu.cont()] = lambda cont, x, y: pu.loss.standard_loss(loss_cont)(cont, self.outcomes, x, y)
            self.entropy_fn[pu.cont()] = lambda y: pu.loss.standard_entropy(loss_cont)(self.quiz_reverse, self.outcomes, y)

        if isinstance(loss_quiz, FunctionType):
            self.loss_fn[pu.quiz()] = lambda cont, x, y: loss_quiz(cont, self.outcomes, x, y)
            self.entropy_fn[pu.quiz()] = NotImplemented
        elif isinstance(loss_quiz, pu.Loss):
            self.loss_fn[pu.quiz()] = lambda cont, x, y: pu.loss.standard_loss(loss_quiz)(cont, self.outcomes, x, y)
            self.entropy_fn[pu.quiz()] = lambda y: pu.loss.standard_entropy(loss_quiz)(self.quiz_reverse, self.outcomes, y)

    @property
    def quiz(self) -> pu.YgivenX:
        return self._quiz

    @quiz.setter
    def quiz(self, value: np.ndarray):
        self._quiz = self.strategy.to_quiz_strategy(value)
        self.marginal_message = self.strategy.update_message_marginal()
        self.quiz_reverse = self.strategy.update_strategy_quiz_reverse()
        if not self.strategy.is_quiz_legal():
            raise ValueError("not a valid quiz strategy")

    @property
    def cont(self) -> pu.XgivenY:
        return self._cont

    @cont.setter
    def cont(self, value: np.ndarray):
        self._cont = self.strategy.to_cont_strategy(value)
        if not self.strategy.is_cont_legal():
            raise ValueError("not a valid quiz strategy")

    def play(self, actions: Dict[pu.Agent, np.ndarray]) -> Dict[pu.Agent, float]:
        self.cont = actions[pu.cont()]
        self.quiz = actions[pu.quiz()]

        return {
            pu.cont(): self.get_expected_loss(pu.cont()),
            pu.quiz(): self.get_expected_loss(pu.quiz())
        }

    def get_expected_loss(self, agent: pu.Agent) -> float:
        loss: float = 0
        for x in self.outcomes:
            for y in self.messages:
                _l = self.marginal_outcome[x] * self.quiz[x][y] * self.loss_fn[agent](self.cont, x, y)
                if not math.isnan(_l):
                    loss += _l

        # TODO: checken of dit oke is
        return np.sign(loss) * pu.inf_loss if math.isinf(loss) else loss

    def get_expected_entropy(self, agent: pu.Agent) -> Optional[float]:
        if callable(self.entropy_fn[agent]):
            ent: float = 0
            for y in self.messages:
                e = self.marginal_message[y] * self.entropy_fn[agent](y)
                if not math.isnan(e):
                    ent += e

            return ent

        return None

    def reset(self):
        pass
        # self._quiz = {x: {y: 0 for y in self.messages} for x in self.outcomes}
        # self.quiz_reverse = {y: {x: 0 for x in self.outcomes} for y in self.messages}
        # self._cont = {y: {x: 0 for x in self.outcomes} for y in self.messages}
        # self.marginal_message = {y: 0 for y in self.messages}

    def get_cont_readable(self):
        return {y.id: {x.id: self.cont[y][x] for x in self.outcomes} for y in self.messages}

    def get_quiz_readable(self):
        return {x.id: {y.id: self.quiz[x][y] for y in self.messages} for x in self.outcomes}
