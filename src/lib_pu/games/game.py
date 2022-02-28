from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

import numpy as np

import src.lib_pu as pu

from prettytable import PrettyTable
from functools import partial

import src.lib_pu.games as pu_games


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
    entropy_cont = None
    matrix = {}

    def __init__(self, loss_names: Dict[pu.Agent, str], matrix: Optional[Dict[pu.Agent], np.ndarray] = None, random_outcome_dist: bool = False):
        outcomes, messages = self.create_structure(len(self.default_outcome_dist()), self.message_structure())

        if random_outcome_dist:
            outcome_dist = dict(zip(outcomes, self.sample_categorical_distribution(len(outcomes))))
        else:
            outcome_dist = dict(zip(outcomes, self.default_outcome_dist()))

        self.strategy_util = pu.StrategyUtil(self)

        self.outcomes = outcomes
        self.messages = messages
        self.outcome_dist = outcome_dist
        self.loss_names = loss_names
        
        if loss_names[pu.CONT].startswith(pu.MATRIX):
            self.matrix = matrix
            self.loss = {agent: partial(pu.LOSS_FNS[pu.MATRIX], self.matrix[agent]) for agent in pu.AGENTS}
            self.entropy_cont = partial(pu.ENTROPY_FNS[pu.MATRIX], self.matrix[pu.CONT])
        else:
            self.loss = {agent: pu.LOSS_FNS[loss_names[agent]] for agent in pu.AGENTS}
            self.entropy_cont = pu.ENTROPY_FNS[loss_names[pu.CONT]]
        
        self.action = {agent: None for agent in pu.AGENTS}

    def set_action(self, agent: pu.Agent, value: np.ndarray):
        try:
            self.action[agent] = pu.Action.get_class(agent).from_array(value, self.outcomes, self.messages)
        except IndexError:
            raise pu.InvalidStrategyError(value, self.get_action_shape(agent))

        if agent == pu.HOST:
            self.message_dist = self.strategy_util.get_message_dist()
            self.host_reverse = self.strategy_util.get_host_reverse()

    def step(self, actions: Dict[str, np.ndarray]) -> Dict[pu.Agent, float]:
        for agent in pu.AGENTS:
            self.set_action(agent, actions[agent])

        return {agent: self.get_expected_loss(agent) for agent in pu.AGENTS}

    def get_expected_loss(self, agent: pu.Agent) -> float:
        loss: float = 0
        for x in self.outcomes:
            for y in self.messages:
                l = self.get_loss(agent, x, y)
                P_y_x = self.action[pu.HOST][x, y]
                p_x = self.outcome_dist[x]
                _l = self.outcome_dist[x] * self.action[pu.HOST][x, y] * self.get_loss(agent, x, y)
                if not math.isnan(_l):
                    loss += _l

        return np.sign(loss) * pu.CLIPPED_INFINITY_LOSS if math.isinf(loss) else loss

    def get_loss(self, agent: pu.Agent, x: pu.Outcome, y: pu.Message):
        return self.loss[agent](self.action[pu.CONT], self.outcomes, x, y)

    """Returns entorpy for the contestant. In theory we can also calculate this for the host, but it would have no real meaning"""
    def get_expected_entropy(self) -> Optional[float]:
        ent: float = 0
        for y in self.messages:
            e = self.message_dist[y] * self.get_entropy(y)
            if not math.isnan(e):
                ent += e
                
        return ent

    def get_entropy(self, y: pu.Message):
        return self.entropy_cont(self.loss[pu.CONT], self.host_reverse, self.outcomes, y)

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
    
    """Determines the lambda_x vector, given the loss and strategy for the contestant"""
    def get_lambda_x_vector(self) -> Dict[pu.Outcome, float]:
        vec = {}
        for x in self.outcomes:
            vec[x] = max(self.get_loss(pu.CONT, x, y) for y in x.messages)
        return vec
    
    """Checks whether the lambda_x vector determined from contestant's loss and strategy is a KT (Kuhn Tucker) vector"""
    def is_lambda_x_vector_kt(self, vec: Dict[pu.Outcome, float]) -> bool:
        ent_dict = {y: self.get_entropy(y) for y in self.messages}
        for y in self.messages:
            _sum = sum(self.host_reverse[x, y] * vec[x] for x in y.outcomes)
            ent = self.get_entropy(y)
            if math.isclose(self.message_dist[y], 0, rel_tol=1e-2) and (ent < _sum or math.isclose(ent, _sum, rel_tol=1e-2)):
                continue
            if self.message_dist[y] > 0 and math.isclose(ent, _sum, rel_tol=1e-2):
                continue
            else:
                return False
        return True
    
    """Checks whether the contestant (Q) plays an equalizer strategy, i.e. the expected loss of Q does not depend on P"""
    def cont_is_equalizer_strategy(self):
        for x in self.outcomes:
            losses = [self.get_loss(pu.CONT, x, y) for y in x.messages]
            
            for loss in losses:
                if not math.isclose(loss, losses[0], rel_tol=1e-2):
                    return False
        return True
    
    """Checks, for both agents, whether they play a worst-case optimal strategy, according to the lambda_x vector. If the vector is a KT-vector, they play worst-case optimally."""
    def is_worst_case_optimal(self) -> bool:
        return self.is_lambda_x_vector_kt(self.get_lambda_x_vector())
    
    def get_outcome_diffs(self) -> Dict[Tuple[pu.Message, pu.Message], pu.Outcome]:
        diffs = {}
        for y1 in range(len(self.messages)):
            for y2 in range(y1 + 1, len(self.messages)):
                # Determine the difference between message y1 and y2
                diff1 = list(set(self.messages[y1].outcomes) - set(self.messages[y2].outcomes))
                diff2 = list(set(self.messages[y2].outcomes) - set(self.messages[y1].outcomes))
                
                # If y1 and y2 differ by the exchange of one outcome:
                if len(diff1) == 1 and len(diff2) == 1 and diff1[0] != diff2[0]:
                    diffs[(self.messages[y1], self.messages[y2])] = diff1[0]
                    diffs[(self.messages[y2], self.messages[y1])] = diff2[0]
        return diffs
    
    """Check whether a matrix is symmetric w.r.t. exchanges in message set Y. The loss function does not need to be fully symmetric in order for the theorems of (van Ommen et. al. 2015) to hold."""
    def is_matrix_symmetric_with_respect_to_exchanges(self, agent) -> bool:
        m = self.matrix[agent]
        for y1 in range(len(self.messages)):
            for y2 in range(y1 + 1, len(self.messages)):
                # Determine the difference between message y1 and y2
                diff1 = list(set(self.messages[y1].outcomes) - set(self.messages[y2].outcomes))
                diff2 = list(set(self.messages[y2].outcomes) - set(self.messages[y1].outcomes))
                
                # If y1 and y2 differ by the exchange of one outcome:
                if len(diff1) == 1 and len(diff2) == 1 and diff1[0] != diff2[0]:
                    x1 = diff1[0].id
                    x2 = diff2[0].id
                    if m[x1, x1] != m[x2, x2]:
                        return False
                    if m[x1, x2] != m[x2, x1]:
                        return False
                    for x_prime in range(len(self.outcomes)):
                        if x_prime == x1 or x_prime == x2:
                            continue
                        if m[x_prime, x1] != m[x_prime, x2]:
                            return False
                        if m[x1, x_prime] != m[x2, x_prime]:
                            return False
        return True
    
    def is_matrix_fully_symmetric(self, agent) -> bool:
        m = self.matrix[agent]
        for x1 in range(len(self.outcomes)):
            for x2 in range(x1 + 1, len(self.outcomes)):
                if m[x1, x1] != m[x2, x2]:
                    return False
                if m[x1, x2] != m[x2, x1]:
                    return False
                for x_prime in range(len(self.outcomes)):
                    if x_prime == x1 or x_prime == x2:
                        continue
                    if m[x_prime, x1] != m[x_prime, x2]:
                        return False
                    if m[x1, x_prime] != m[x2, x_prime]:
                        return False
        return True

    def get_filtered_action(self, agent: pu.Agent):
        action = {}
        if agent == pu.CONT:
            for y in self.messages:
                if len(y.outcomes) < 2:
                    continue
                if y not in action:
                    action[y] = {}
                for x in y.outcomes:
                    action[y][x] = self.action[agent][x, y]
        elif agent == pu.HOST:
            for x in self.outcomes:
                if len(x.messages) < 2:
                    continue
                if x not in action:
                    action[x] = {}
                for y in x.messages:
                    action[x][y] = self.action[agent][x, y]
        return action

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
    
    @classmethod
    def pretty_name(cls) -> str:
        return pu_games.GAME_PRETTY_NAMES[cls.name()]

    @staticmethod
    @abstractmethod
    def default_outcome_dist() -> List[float]:
        pass

    @staticmethod
    @abstractmethod
    def message_structure() -> List[List[int]]:
        pass
    
    @classmethod
    def get_outcome_count(cls) -> int:
        return len(cls.default_outcome_dist())

    @classmethod
    def get_message_count(cls) -> int:
        return len(cls.message_structure())

    @staticmethod
    @abstractmethod
    def cont_default() -> np.ndarray:
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
            table.add_row([y, "p=" + '{:.3f}'.format(self.message_dist[y]) + " | " + str([str(x) for x in y.outcomes]).translate({39: None}).strip('[]')])
        
        table.add_row(['', ''])
        table.add_row(['Message differences', ''])
        diffs = self.get_outcome_diffs()
        for y1, y2 in diffs:
            table.add_row([str((y1, y2)), diffs[(y1, y2)]])

        table.add_row(['', ''])
        table.add_row(['Cont loss', self.loss_names[pu.CONT]])
        if self.loss_names[pu.CONT].startswith(pu.MATRIX):
            col_maxes = max([max([len("{:g}".format(x)) for x in col]) for col in self.matrix[pu.CONT].T], [max([len("{:g}".format(x)) for x in col]) for col in self.matrix[pu.HOST].T])
            for x in self.matrix[pu.CONT]:
                _row = ''
                for i, y in enumerate(x):
                    _row += ("{:"+str(col_maxes[i])+"g} ").format(y)
                table.add_row(['', _row])
            table.add_row(['Symmetric w.r.t. exchanges?', self.is_matrix_symmetric_with_respect_to_exchanges(pu.CONT)])
                
        table.add_row(['Host loss', self.loss_names[pu.HOST]])
        if self.loss_names[pu.HOST].startswith(pu.MATRIX):
            col_maxes = max([max([len("{:g}".format(x)) for x in col]) for col in self.matrix[pu.CONT].T], [max([len("{:g}".format(x)) for x in col]) for col in self.matrix[pu.HOST].T])
            for x in self.matrix[pu.HOST]:
                _row = ''
                for i, y in enumerate(x):
                    _row += ("{:"+str(col_maxes[i])+"g} ").format(y)
                table.add_row(['', _row])
            table.add_row(['Symmetric w.r.t. exchanges?', self.is_matrix_symmetric_with_respect_to_exchanges(pu.HOST)])

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
            table.add_row(['Expected entropy', self.get_expected_entropy()])
        except Exception as e:
            table.add_row(['Cont action', 'ERROR'])
            
        table.add_row(['', ''])
        table.add_row(['Host reverse action', ''])

        for y in self.messages:
            for x in self.outcomes:
                if x == self.outcomes[0]:
                    table.add_row([y, f"{x}: {self.host_reverse[x, y]}"])
                else:
                    table.add_row(['', f"{x}: {self.host_reverse[x, y]}"])

        table.add_row(['', ''])
        table.add_row(['Host action', ''])
        table.add_row(['Action space', self.get_action_shape(pu.HOST)])
        
        for x in self.outcomes:
            for y in self.messages:
                if y == self.messages[0]:
                    table.add_row([x, f"{y}: {self.action[pu.HOST][x, y]}"])
                else:
                    table.add_row(['', f"{y}: {self.action[pu.HOST][x, y]}"])

        table.add_row(['Host expected loss', self.get_expected_loss(pu.HOST)])
        
        table.add_row(['', ''])
        table.add_row(['RCAR dist: ', "{:.3f}".format(self.strategy_util.rcar_dist())])
        table.add_row(['Is P RCAR?', self.strategy_util.is_rcar()])
        
        table.add_row(['', ''])
        table.add_row(['Lambda_x vector:', self.get_lambda_x_vector()])
        table.add_row(['Cont equalizer?', self.cont_is_equalizer_strategy()])
        table.add_row(['Worst-case optimal?', self.is_worst_case_optimal()])
        
        return str(table)
