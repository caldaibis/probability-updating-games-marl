from __future__ import annotations

import numpy as np
import random
from abc import abstractmethod
from statistics import mean
from typing import List, Dict, Union

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

    loss: pu.Loss
    _loss_type: pu.LossType

    strategy: pu.Strategy
    _quiz: pu.YgivenX
    quiz_reverse: pu.XgivenY
    _cont: pu.XgivenY

    def __init__(self, name: str, outcomes: List[pu.Outcome], messages: List[pu.Message], marginal_outcome: Dict[pu.Outcome, float]):
        self.strategy = pu.Strategy(self)
        self.loss = pu.Loss(self)

        self._name = name
        self.outcomes = outcomes
        self.messages = messages
        self.marginal_outcome = marginal_outcome

    @property
    @abstractmethod
    def name(self) -> str:
        return self._name

    @property
    def loss_type(self) -> pu.LossType:
        return self._loss_type

    @loss_type.setter
    def loss_type(self, value: pu.LossType):
        self._loss_type = value

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

    def play(self, actions: Union[Dict[pu.Agent, pu.PreStrategy], Dict[pu.Agent, np.ndarray]]) -> Dict[pu.Agent, float]:
        if isinstance(actions[pu.quiz()], pu.PreStrategy):
            self.quiz = actions[pu.quiz()]
            self.cont = actions[pu.cont()]
        else:
            self.quiz = self.strategy.to_pre_quiz_strategy(actions[pu.quiz()])
            self.cont = self.strategy.to_pre_cont_strategy(actions[pu.cont()])

        reward_quiz = -self.loss.get_expected_loss()
        reward_cont = self.loss.get_expected_loss()

        return {pu.quiz(): reward_quiz, pu.cont(): reward_cont}

    def simulate_single(self) -> (pu.Outcome, pu.Message, float, float):
        x = random.choices(list(self.marginal_outcome.keys()), list(self.marginal_outcome.values()), k=1)[0]
        y = random.choices(list(self.quiz[x].keys()), list(self.quiz[x].values()), k=1)[0]
        loss = self.loss.get_loss(x, y)
        entropy = self.loss.get_entropy(y)

        return x, y, loss, entropy

    def simulate(self, n: int) -> (Dict[pu.Outcome, int], Dict[pu.Message, int], float, float):
        x_count = {x: 0 for x in self.outcomes}
        y_count = {y: 0 for y in self.messages}
        losses = []
        entropies = []

        for _ in range(n):
            x, y, loss, entropy = self.simulate_single()
            x_count[x] += 1
            y_count[y] += 1
            losses.append(loss)
            entropies.append(entropy)

        return x_count, y_count, mean(losses), mean(entropies)
