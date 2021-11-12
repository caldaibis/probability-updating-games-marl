from __future__ import annotations

from types import FunctionType
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Optional

import numpy as np

import probability_updating as pu


class Game(ABC):
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

    def __init__(self, losses: Dict[pu.Agent, pu.LossFunc | pu.Loss]):
        outcomes, messages = self.create_structure(self.marginal(), self.message_structure())
        marginal_outcome = dict(zip(outcomes, self.marginal()))

        self.strategy = pu.Strategy(self)

        self.outcomes = outcomes
        self.messages = messages
        self.marginal_outcome = marginal_outcome

        if isinstance(losses[pu.cont()], FunctionType):
            self.loss_fn[pu.cont()] = lambda cont, x, y: losses[pu.cont()](cont, self.outcomes, x, y)
            self.entropy_fn[pu.cont()] = NotImplemented
        elif isinstance(losses[pu.cont()], pu.Loss):
            self.loss_fn[pu.cont()] = lambda cont, x, y: pu.loss.standard_loss(losses[pu.cont()])(cont, self.outcomes, x, y)
            self.entropy_fn[pu.cont()] = lambda y: pu.loss.standard_entropy(losses[pu.cont()])(self.quiz_reverse, self.outcomes, y)

        if isinstance(losses[pu.quiz()], FunctionType):
            self.loss_fn[pu.quiz()] = lambda cont, x, y: losses[pu.quiz()](cont, self.outcomes, x, y)
            self.entropy_fn[pu.quiz()] = NotImplemented
        elif isinstance(losses[pu.quiz()], pu.Loss):
            self.loss_fn[pu.quiz()] = lambda cont, x, y: pu.loss.standard_loss(losses[pu.quiz()])(cont, self.outcomes, x, y)
            self.entropy_fn[pu.quiz()] = lambda y: pu.loss.standard_entropy(losses[pu.quiz()])(self.quiz_reverse, self.outcomes, y)

    @property
    def quiz(self) -> pu.YgivenX:
        return self._quiz

    @quiz.setter
    def quiz(self, value: np.ndarray):
        try:
            self._quiz = self.strategy.to_quiz_strategy(value)
        except IndexError:
            raise ValueError("Not a valid quiz strategy. Did you accidentally swap the cont and quiz strategies?")
        self.marginal_message = self.strategy.update_message_marginal()
        self.quiz_reverse = self.strategy.update_strategy_quiz_reverse()

        assert self.strategy.is_quiz_legal()

    @property
    def cont(self) -> pu.XgivenY:
        return self._cont

    @cont.setter
    def cont(self, value: np.ndarray):
        try:
            self._cont = self.strategy.to_cont_strategy(value)
        except IndexError:
            raise ValueError("Not a valid cont strategy. Did you accidentally swap the cont and quiz strategies?")

        assert self.strategy.is_cont_legal()

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

    def get_cont_readable(self):
        return {y.id: {x.id: self.cont[y][x] for x in self.outcomes} for y in self.messages}

    def get_quiz_readable(self):
        return {x.id: {y.id: self.quiz[x][y] for y in self.messages} for x in self.outcomes}

    def get_cont_action_space(self) -> int:
        return sum(len(y.outcomes) - 1 for y in self.messages)

    def get_quiz_action_space(self) -> int:
        return sum(len(x.messages) - 1 for x in self.outcomes)

    @staticmethod
    def create_structure(marginal: List[float], messages: List[List[int]]) -> (List[pu.Outcome], List[pu.Message]):
        # create messages
        new_messages: List[pu.Message] = []
        for y in range(len(messages)):
            new_messages.append(pu.Message(y, []))

        # create outcomes
        new_outcomes: List[pu.Outcome] = []
        for x in range(len(marginal)):
            new_outcomes.append(pu.Outcome(x, []))

        # fill messages
        for y in range(len(messages)):
            for x in messages[y]:
                new_messages[y].outcomes.append(new_outcomes[x])

        # infer outcome structure from outcome count and message structure
        outcomes: List[List[int]] = []
        for x in range(len(marginal)):
            ys: List[int] = []
            for y in range(len(messages)):
                if x in messages[y]:
                    ys.append(y)
            outcomes.append(ys)

        # fill outcomes
        for x in range(len(outcomes)):
            for y in outcomes[x]:
                new_outcomes[x].messages.append(new_messages[y])

        return new_outcomes, new_messages

    @staticmethod
    @abstractmethod
    def name():
        pass

    @staticmethod
    @abstractmethod
    def marginal() -> List[float]:
        pass

    @staticmethod
    @abstractmethod
    def message_structure() -> List[List[int]]:
        pass
