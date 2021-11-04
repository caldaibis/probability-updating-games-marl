from __future__ import annotations

import math
from abc import abstractmethod
from typing import List, Dict, Callable

import numpy as np

import probability_updating as pu


def create_structure(marginal: List[float], messages: List[List[int]]) -> (List[pu.Outcome], List[pu.Message]):
    # create messages
    new_messages: List[pu.Message] = []
    for y in range(len(messages)):
        new_messages.append(pu.Message(y))

    # create outcomes
    new_outcomes: List[pu.Outcome] = []
    for x in range(len(marginal)):
        new_outcomes.append(pu.Outcome(x))

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


class Game:
    _name: str

    outcomes: List[pu.Outcome]
    messages: List[pu.Message]

    marginal_outcome: Dict[pu.Outcome, float]
    marginal_message: Dict[pu.Message, float]

    _loss_cont_fn: Callable[[pu.Outcome, pu.Message], float]
    _loss_quiz_fn: Callable[[pu.Outcome, pu.Message], float]

    _entropy_cont_fn: Callable[[pu.Message], float]
    _entropy_quiz_fn: Callable[[pu.Message], float]

    strategy: pu.Strategy
    _quiz: pu.YgivenX
    quiz_reverse: pu.XgivenY
    _cont: pu.XgivenY

    def __init__(self,
                 name: str,
                 outcomes: List[pu.Outcome],
                 messages: List[pu.Message],
                 marginal_outcome: Dict[pu.Outcome, float],
                 loss_cont_fn: pu.LossFunc,
                 loss_quiz_fn: pu.LossFunc,
                 entropy_cont_fn: pu.): # koppel entropy function met loss!!
        self.strategy = pu.Strategy(self)

        self._name = name
        self.outcomes = outcomes
        self.messages = messages
        self.marginal_outcome = marginal_outcome

        self._loss_cont_fn = lambda x, y: loss_cont_fn(self.cont, self.outcomes, x, y)
        self._loss_quiz_fn = lambda x, y: loss_quiz_fn(self.cont, self.outcomes, x, y)

        entropy_cont_fn = pu.get_entropy_fn(loss_cont_fn.__name__)
        self._entropy_cont_fn = lambda y: entropy_cont_fn(self.quiz_reverse, self.outcomes, y)

        entropy_quiz_fn = pu.get_entropy_fn(loss_quiz_fn.__name__)
        self._entropy_quiz_fn = lambda y: entropy_quiz_fn(self.quiz_reverse, self.outcomes, y)

    @property
    @abstractmethod
    def name(self) -> str:
        return self._name

    @property
    def quiz(self) -> pu.YgivenX:
        return self._quiz

    @quiz.setter
    def quiz(self, value: pu.PreStrategy):
        self._quiz = self.strategy.to_quiz_strategy(value)
        self.marginal_message = self.strategy.update_message_marginal()
        self.quiz_reverse = self.strategy.update_strategy_quiz_reverse()
        if not self.strategy.is_quiz_legal():
            raise ValueError("not a valid quiz strategy")

    @property
    def cont(self) -> pu.XgivenY:
        return self._cont

    @cont.setter
    def cont(self, value: pu.PreStrategy):
        self._cont = self.strategy.to_cont_strategy(value)
        if not self.strategy.is_cont_legal():
            raise ValueError("not a valid quiz strategy")

    def play(self, actions: Dict[pu.Agent, pu.PreStrategy] | Dict[pu.Agent, np.ndarray]) -> Dict[pu.Agent, float]:
        if isinstance(actions[pu.quiz()], pu.PreStrategy):
            self.quiz = actions[pu.quiz()]
            self.cont = actions[pu.cont()]
        else:
            self.quiz = self.strategy.to_pre_quiz_strategy(actions[pu.quiz()])
            self.cont = self.strategy.to_pre_cont_strategy(actions[pu.cont()])

        reward_quiz = self._get_expected_loss()
        reward_cont = -self._get_expected_loss()

        return {pu.quiz(): reward_quiz, pu.cont(): reward_cont}

    def _get_expected_loss(self) -> float:
        loss: float = 0
        for x in self.outcomes:
            for y in self.messages:
                _l = self.marginal_outcome[x] * self.quiz[x][y] * self._loss_fn(x, y)
                if not math.isnan(_l):
                    loss += _l

        return 100000000000 if math.isinf(loss) else loss

    def _get_expected_entropy(self) -> float:
        ent: float = 0
        for y in self.messages:
            e = self.marginal_message[y] * self.get_entropy(y)
            if not math.isnan(e):
                ent += e

        return ent