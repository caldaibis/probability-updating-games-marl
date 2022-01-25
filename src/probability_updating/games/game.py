from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np

import probability_updating as pu
from exceptions import *

from prettytable import PrettyTable


class Game(ABC):
    outcomes: List[pu.Outcome]
    messages: List[pu.Message]

    marginal_outcome: Dict[pu.Outcome, float]
    marginal_message: Dict[pu.Message, float]

    loss: Dict[pu.Agent, pu.Loss]
    entropy: Dict[pu.Agent, Optional[pu.EntropyFunc]]

    strategy_util: pu.StrategyUtil
    action: Dict[pu.Agent, pu.Action]
    host_reverse: pu.ContAction

    def __init__(self, losses: Dict[pu.Agent, pu.Loss], random_marginal: bool = False):
        outcomes, messages = self.create_structure(len(self.default_marginal()), self.message_structure())

        if random_marginal:
            marginal_outcome = dict(zip(outcomes, pu.random_marginal_distribution(len(outcomes))))
        else:
            marginal_outcome = dict(zip(outcomes, self.default_marginal()))

        self.strategy_util = pu.StrategyUtil(self)

        self.outcomes = outcomes
        self.messages = messages
        self.marginal_outcome = marginal_outcome

        self.loss = losses
        self.entropy = {agent: pu.get_entropy_fn(losses[agent]) for agent in pu.Agent}

        self.action = {agent: None for agent in pu.Agent}

    def set_action(self, agent: pu.Agent, value: np.ndarray):
        try:
            self.action[agent] = pu.Action.get_class(agent).from_array(value, self.outcomes, self.messages)
        except IndexError:
            raise InvalidStrategyError(value, self.get_action_space(agent))

        if agent == pu.Agent.Host:
            self.marginal_message = self.strategy_util.update_message_marginal()
            self.host_reverse = self.strategy_util.update_strategy_host_reverse()

    def step(self, actions: Dict[str, np.ndarray]) -> Dict[pu.Agent, float]:
        for agent in pu.Agent:
            self.set_action(agent, actions[agent.value])

        return self.get_expected_losses()

    def any_illegal_action(self) -> bool:
        cont_out_domain = not self.action[pu.Agent.Cont].is_within_domain(self.outcomes, self.messages)
        host_out_domain = not self.action[pu.Agent.Host].is_within_domain(self.outcomes, self.messages)

        cont_no_distribution = not self.action[pu.Agent.Cont].sums_to_one(self.outcomes, self.messages)
        host_no_distribution = not self.action[pu.Agent.Host].sums_to_one(self.outcomes, self.messages)

        return cont_out_domain or host_out_domain or cont_no_distribution or host_no_distribution

    def get_expected_losses(self) -> Dict[pu.Agent, float]:
        if self.any_illegal_action():
            return {agent.value: pu.inf_loss for agent in pu.Agent}

        return {agent.value: self.get_expected_loss(agent) for agent in pu.Agent}

    def get_expected_loss(self, agent: pu.Agent) -> float:
        loss: float = 0
        for x in self.outcomes:
            for y in self.messages:
                _l = self.marginal_outcome[x] * self.action[pu.Agent.Host][x, y] * self.get_loss(agent, x, y)
                if not math.isnan(_l):
                    loss += _l

        return np.sign(loss) * pu.inf_loss if math.isinf(loss) else loss

    def get_loss(self, agent: pu.Agent, x: pu.Outcome, y: pu.Message):
        return self.loss[agent](self.action[pu.Agent.Cont], self.outcomes, x, y)

    def get_expected_entropies(self):
        if self.any_illegal_action():
            return {agent.value: pu.inf_loss for agent in pu.Agent}

        return {agent.value: self.get_expected_entropy(agent) for agent in pu.Agent}

    def get_expected_entropy(self, agent: pu.Agent) -> Optional[float]:
        ent: float = 0
        for y in self.messages:
            e = self.marginal_message[y] * self.get_entropy(agent, y)
            if not math.isnan(e):
                ent += e

        return ent

    def get_entropy(self, agent: pu.Agent, y: pu.Message):
        if self.entropy[agent]:
            return self.entropy[agent](self.host_reverse, self.outcomes, y)

        return math.nan

    def get_action_space(self, agent: pu.Agent):
        if agent == pu.Agent.Cont:
            return sum(len(y.outcomes) - 1 for y in self.messages)
        elif agent == pu.Agent.Host:
            return sum(len(x.messages) - 1 for x in self.outcomes)

    def is_graph_game(self) -> bool:
        return all(len(y.outcomes) <= 2 for y in self.messages)

    def is_matroid_game(self) -> bool:
        # each outcome x must occur in some message y
        if not all(len(x.messages) > 0 for x in self.outcomes):
            return False

        # basis exchange property (hopefully correct?)
        for y1 in self.messages:
            for y2 in self.messages:
                if y1 == y2:
                    continue
                for x1 in y1.outcomes:
                    if x1 in y2.outcomes:
                        continue
                    true_for_some = False
                    for x2 in y2.outcomes:
                        if x2 in y1.outcomes:
                            continue
                        outcome_list = y1.outcomes.copy()
                        outcome_list.remove(x1)
                        outcome_list.append(x2)
                        for y in self.messages:
                            if set(outcome_list) == set(y.outcomes):
                                true_for_some = True
                    if not true_for_some:
                        return False

        return True

    @staticmethod
    def create_structure(outcome_count: int, messages: List[List[int]]) -> (List[pu.Outcome], List[pu.Message]):
        # create messages
        new_messages: List[pu.Message] = []
        for y in range(len(messages)):
            new_messages.append(pu.Message(y, []))

        # create outcomes
        new_outcomes: List[pu.Outcome] = []
        for x in range(outcome_count):
            new_outcomes.append(pu.Outcome(x, []))

        # fill messages
        for y in range(len(messages)):
            for x in messages[y]:
                new_messages[y].outcomes.append(new_outcomes[x])

        # infer outcome structure from outcome count and message structure
        outcomes: List[List[int]] = []
        for x in range(outcome_count):
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
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def default_marginal() -> List[float]:
        pass

    @staticmethod
    @abstractmethod
    def message_structure() -> List[List[int]]:
        pass

    @staticmethod
    @abstractmethod
    def cont_optimal_zero_one() -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def host_default() -> np.ndarray:
        pass

    def __str__(self):
        table = PrettyTable(['Game', self.name()], align="l")

        table.add_row(['Graph game?', self.is_graph_game()])
        table.add_row(['Matroid game?', self.is_matroid_game()])
        table.add_row(['', ''])

        for x in self.outcomes:
            table.add_row([x, "p=" + '{:.3f}'.format(self.marginal_outcome[x]) + " | " + str([str(y) for y in x.messages]).translate({39: None}).strip('[]')])

        table.add_row(['', ''])
        for y in self.messages:
            try:
                table.add_row([y, "p=" + '{:.3f}'.format(self.marginal_message[y]) + " | " + str([str(x) for x in y.outcomes]).translate({39: None}).strip('[]')])
            except AttributeError:
                table.add_row([y, "p=?     | " + str([str(x) for x in y.outcomes]).translate({39: None}).strip('[]')])

        table.add_row(['', ''])
        table.add_row(['Cont loss', self.loss[pu.Agent.Cont].name])
        table.add_row(['Host loss', self.loss[pu.Agent.Host].name])

        table.add_row(['', ''])
        table.add_row(['Cont action', ''])
        table.add_row(['Action space', self.get_action_space(pu.Agent.Cont)])

        try:
            for y in self.messages:
                for x in self.outcomes:
                    if x == self.outcomes[0]:
                        table.add_row([y, f"{x}: {self.action[pu.Agent.Cont][x, y]}"])
                    else:
                        table.add_row(['', f"{x}: {self.action[pu.Agent.Cont][x, y]}"])

            table.add_row(['Cont expected loss', self.get_expected_losses()[pu.Agent.Cont.value]])
            table.add_row(['Cont expected entropy', self.get_expected_entropies()[pu.Agent.Cont.value]])
        except AttributeError:
            table.add_row(['Cont action', None])

        table.add_row(['', ''])
        table.add_row(['Host action', ''])
        table.add_row(['Action space', self.get_action_space(pu.Agent.Host)])
        try:
            for x in self.outcomes:
                for y in self.messages:
                    if y == self.messages[0]:
                        table.add_row([x, f"{y}: {self.action[pu.Agent.Host][x, y]}"])
                    else:
                        table.add_row(['', f"{y}: {self.action[pu.Agent.Host][x, y]}"])

            table.add_row(['CAR?', self.strategy_util.is_car()])
            table.add_row(['RCAR?', self.strategy_util.is_rcar()])
            table.add_row(['RCAR RMSE: ', self.strategy_util.rcar_dist()])
            table.add_row(['Host expected loss', self.get_expected_losses()[pu.Agent.Host.value]])
            table.add_row(['Host expected entropy', self.get_expected_entropies()[pu.Agent.Host.value]])
        except AttributeError:
            table.add_row(['Host action', None])

        return str(table)
