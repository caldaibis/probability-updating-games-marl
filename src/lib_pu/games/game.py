from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np

import src.lib_pu as pu

from prettytable import PrettyTable
from functools import partial


class Game(ABC):
    outcomes: List[pu.Outcome]
    messages: List[pu.Message]

    outcome_dist: Dict[pu.Outcome, float]
    message_dist: Dict[pu.Message, float]

    strategy_util: pu.StrategyUtil
    action: Dict[pu.Agent, pu.Action]
    host_reverse: pu.ContAction
    
    loss = {}
    loss_names = {}
    entropy = {}
    matrix = {}

    def __init__(self, loss_names: Dict[pu.Agent, str], matrix_gen: Optional[Dict[pu.Agent], np.ndarray] = None, random_outcome_dist: bool = False):
        outcomes, messages = self.create_structure(len(self.default_outcome_dist()), self.message_structure())

        if random_outcome_dist:
            outcome_dist = dict(zip(outcomes, self.sample_categorical_distribution(len(outcomes))))
        else:
            outcome_dist = dict(zip(outcomes, self.default_outcome_dist()))

        self.strategy_util = pu.StrategyUtil(self)

        self.outcomes = outcomes
        self.messages = messages
        self.outcome_dist = outcome_dist

        if matrix_gen:
            self.matrix = {agent: matrix_gen[agent](len(outcomes)) for agent in matrix_gen}

        self.loss_names = loss_names
        
        if loss_names[pu.CONT] == pu.MATRIX:
            self.loss = {agent: partial(pu.LOSS_FN_LIST[loss_names[agent]], self.matrix[agent]) for agent in pu.AGENTS}
        else:
            self.loss = {agent: pu.LOSS_FN_LIST[loss_names[agent]] for agent in pu.AGENTS}
            
        self.entropy = {agent: pu.ENTROPY_FN_LIST[loss_names[agent]] for agent in pu.AGENTS}

        self.action = {agent: None for agent in pu.AGENTS}

    def set_action(self, agent: pu.Agent, value: np.ndarray):
        try:
            self.action[agent] = pu.Action.get_class(agent).from_array(value, self.outcomes, self.messages)
        except IndexError:
            raise pu.InvalidStrategyError(value, self.get_action_shape(agent))

        if agent == pu.HOST:
            self.message_dist = self.strategy_util.update_message_dist()
            self.host_reverse = self.strategy_util.update_strategy_host_reverse()

    def step(self, actions: Dict[str, np.ndarray]) -> Dict[pu.Agent, float]:
        for agent in pu.AGENTS:
            self.set_action(agent, actions[agent])

        return {agent: self.get_expected_loss(agent) for agent in pu.AGENTS}

    def get_expected_loss(self, agent: pu.Agent) -> float:
        loss: float = 0
        for x in self.outcomes:
            for y in self.messages:
                _l = self.outcome_dist[x] * self.action[pu.HOST][x, y] * self.get_loss(agent, x, y)
                if not math.isnan(_l):
                    loss += _l

        return np.sign(loss) * pu.CLIPPED_INFINITY_LOSS if math.isinf(loss) else loss

    def get_loss(self, agent: pu.Agent, x: pu.Outcome, y: pu.Message):
        return self.loss[agent](self.action[pu.CONT], self.outcomes, x, y)

    def get_expected_entropy(self, agent: pu.Agent) -> Optional[float]:
        ent: float = 0
        for y in self.messages:
            e = self.message_dist[y] * self.get_entropy(agent, y)
            if not math.isnan(e):
                ent += e
                
        return ent

    def get_entropy(self, agent: pu.Agent, y: pu.Message):
        return self.entropy[agent](self.loss[agent], self.host_reverse, self.outcomes, y)

    def get_action_shape(self, agent: pu.Agent) -> List[int]:
        shape = []
        if agent == pu.CONT:
            for y in self.messages:
                if len(y.outcomes) > 1:
                    shape.append(len(y.outcomes))
        elif agent == pu.HOST:
            for x in self.outcomes:
                if len(x.messages) > 1:
                    shape.append(len(x.messages))
        return shape

    def is_graph_game(self) -> bool:
        return all(len(y.outcomes) <= 2 for y in self.messages)

    def is_matroid_game(self) -> bool:
        # each outcome x must occur in some message y
        if not all(len(x.messages) > 0 for x in self.outcomes):
            return False

        # basis exchange property
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
    def sample_categorical_distribution(outcome_count: int) -> List[float]:
        """Samples a categorical/discrete distribution, uniform randomly."""
        return np.random.dirichlet([1] * outcome_count).tolist()

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass
    
    @staticmethod
    @abstractmethod
    def pretty_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def default_outcome_dist() -> List[float]:
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

    @staticmethod
    def matprint(mat, fmt="g"):
        pass
        
    def __str__(self):
        table = PrettyTable(['Game', self.name()], align="l")

        table.add_row(['Graph game?', self.is_graph_game()])
        table.add_row(['Matroid game?', self.is_matroid_game()])
        table.add_row(['', ''])

        for x in self.outcomes:
            table.add_row([x, "p=" + '{:.3f}'.format(self.outcome_dist[x]) + " | " + str([str(y) for y in x.messages]).translate({39: None}).strip('[]')])

        table.add_row(['', ''])
        for y in self.messages:
            try:
                table.add_row([y, "p=" + '{:.3f}'.format(self.message_dist[y]) + " | " + str([str(x) for x in y.outcomes]).translate({39: None}).strip('[]')])
            except AttributeError:
                table.add_row([y, "p=?     | " + str([str(x) for x in y.outcomes]).translate({39: None}).strip('[]')])

        table.add_row(['', ''])
        table.add_row(['Cont loss', self.loss_names[pu.CONT]])
        if self.loss_names[pu.CONT] == pu.MATRIX:
            col_maxes = max([max([len("{:g}".format(x)) for x in col]) for col in self.matrix[pu.CONT].T], [max([len("{:g}".format(x)) for x in col]) for col in self.matrix[pu.HOST].T])
            for x in self.matrix[pu.CONT]:
                _row = ''
                for i, y in enumerate(x):
                    _row += ("{:"+str(col_maxes[i])+"g} ").format(y)
                table.add_row(['', _row])
                
        table.add_row(['Host loss', self.loss_names[pu.HOST]])
        if self.loss_names[pu.HOST] == pu.MATRIX:
            col_maxes = max([max([len("{:g}".format(x)) for x in col]) for col in self.matrix[pu.CONT].T], [max([len("{:g}".format(x)) for x in col]) for col in self.matrix[pu.HOST].T])
            for x in self.matrix[pu.HOST]:
                _row = ''
                for i, y in enumerate(x):
                    _row += ("{:"+str(col_maxes[i])+"g} ").format(y)
                table.add_row(['', _row])

        table.add_row(['', ''])
        table.add_row(['Cont action', ''])
        table.add_row(['Action space', self.get_action_shape(pu.CONT)])

        try:
            for y in self.messages:
                for x in self.outcomes:
                    if x == self.outcomes[0]:
                        table.add_row([y, f"{x}: {self.action[pu.CONT][x, y]}"])
                    else:
                        table.add_row(['', f"{x}: {self.action[pu.CONT][x, y]}"])

            table.add_row(['Cont expected loss', self.get_expected_loss(pu.CONT)])
            table.add_row(['Cont expected entropy', self.get_expected_entropy(pu.CONT)])
        except Exception as e:
            table.add_row(['Cont action', 'ERROR'])

        table.add_row(['', ''])
        table.add_row(['Host action', ''])
        table.add_row(['Action space', self.get_action_shape(pu.HOST)])
        try:
            for x in self.outcomes:
                for y in self.messages:
                    if y == self.messages[0]:
                        table.add_row([x, f"{y}: {self.action[pu.HOST][x, y]}"])
                    else:
                        table.add_row(['', f"{y}: {self.action[pu.HOST][x, y]}"])
    
            table.add_row(['CAR?', self.strategy_util.is_car()])
            table.add_row(['RCAR?', self.strategy_util.is_rcar()])
            table.add_row(['RCAR dist: ', self.strategy_util.rcar_dist()])
            table.add_row(['Host expected loss', self.get_expected_loss(pu.HOST)])
            table.add_row(['Host expected entropy', self.get_expected_entropy(pu.HOST)])
        except Exception as e:
            table.add_row(['Host action', 'ERROR'])

        return str(table)
