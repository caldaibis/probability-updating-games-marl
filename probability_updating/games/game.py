from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np

import probability_updating as pu
from exceptions import InvalidStrategyError


class Game(ABC):
    outcomes: List[pu.Outcome]
    messages: List[pu.Message]

    marginal_outcome: Dict[pu.Outcome, float]
    marginal_message: Dict[pu.Message, float]

    loss: Dict[pu.Agent, pu.Loss]
    entropy: Dict[pu.Agent, Optional[pu.EntropyFunc]]

    strategy: pu.Strategy
    _quiz: pu.YgivenX
    quiz_reverse: pu.XgivenY
    _cont: pu.XgivenY

    def __init__(self, losses: Dict[pu.Agent, pu.Loss]):
        outcomes, messages = self.create_structure(self.marginal(), self.message_structure())
        marginal_outcome = dict(zip(outcomes, self.marginal()))

        self.strategy = pu.Strategy(self)

        self.outcomes = outcomes
        self.messages = messages
        self.marginal_outcome = marginal_outcome

        self.loss = {agent: losses[agent] for agent in pu.agents()}
        self.entropy = {agent: pu.get_entropy_fn(losses[agent]) for agent in pu.agents()}

    @property
    def cont(self) -> pu.XgivenY:
        return self._cont

    @cont.setter
    def cont(self, value: np.ndarray):
        try:
            self._cont = self.strategy.to_cont_strategy(value)
        except IndexError:
            raise InvalidStrategyError(value, self.get_cont_action_space())

        assert self.strategy.is_cont_legal()

    @property
    def quiz(self) -> pu.YgivenX:
        return self._quiz

    @quiz.setter
    def quiz(self, value: np.ndarray):
        try:
            self._quiz = self.strategy.to_quiz_strategy(value)
        except IndexError:
            raise InvalidStrategyError(value, self.get_quiz_action_space())

        self.marginal_message = self.strategy.update_message_marginal()
        self.quiz_reverse = self.strategy.update_strategy_quiz_reverse()

        assert self.strategy.is_quiz_legal()

    def play(self, actions: Dict[pu.Agent, np.ndarray]) -> Dict[pu.Agent, float]:
        self.cont = actions[pu.cont()]
        self.quiz = actions[pu.quiz()]

        return {
            pu.cont(): self.get_expected_loss(pu.cont()),
            pu.quiz(): self.get_expected_loss(pu.quiz())
        }

    def get_loss(self, agent: pu.Agent, x: pu.Outcome, y: pu.Message):
        return self.loss[agent](self.cont, self.outcomes, x, y)

    def get_expected_loss(self, agent: pu.Agent) -> float:
        loss: float = 0
        for x in self.outcomes:
            for y in self.messages:
                _l = self.marginal_outcome[x] * self.quiz[x][y] * self.get_loss(agent, x, y)
                if not math.isnan(_l):
                    loss += _l

        return np.sign(loss) * pu.inf_loss if math.isinf(loss) else loss

    def get_entropy(self, agent: pu.Agent, y: pu.Message):
        return self.entropy[agent](self.quiz_reverse, self.outcomes, y)

    def get_expected_entropy(self, agent: pu.Agent) -> Optional[float]:
        if self.entropy[agent]:
            ent: float = 0
            for y in self.messages:
                e = self.marginal_message[y] * self.get_entropy(agent, y)
                if not math.isnan(e):
                    ent += e

            return ent

        return None

    def get_cont_readable(self) -> Dict[int, Dict[int, float]]:
        return {y.id: {x.id: self.cont[y][x] for x in self.outcomes} for y in self.messages}

    def get_quiz_readable(self) -> Dict[int, Dict[int, float]]:
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
