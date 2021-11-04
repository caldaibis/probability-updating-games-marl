from __future__ import annotations

import numpy as np
import math
from typing import Dict
import probability_updating as pu

OutcomeId = int
MessageId = int


class Strategy:
    game: pu.Game

    def __init__(self, game: pu.Game):
        self.game = game

    def update_strategy_quiz_reverse(self) -> XgivenY:
        reverse = {}
        for y in self.game.messages:
            reverse[y] = {}
            for x in self.game.outcomes:
                reverse[y][x] = self.game.quiz[x][y] * self.game.marginal_outcome[x] / self.game.marginal_message[y]

        return reverse

    def update_message_marginal(self) -> Dict[pu.Message, float]:
        probs = {}
        for y in self.game.messages:
            probs[y] = 0
            for x in self.game.outcomes:
                probs[y] += self.game.quiz[x][y] * self.game.marginal_outcome[x]

        return probs

    def is_quiz_legal(self) -> bool:
        is_distribution = math.isclose(sum(self.game.marginal_message.values()), 1, rel_tol=1e-5)

        is_lying = False
        for y in self.game.messages:
            for x in self.game.outcomes:
                if self.game.quiz[x][y] > 0 and x not in y.outcomes:
                    is_lying = True

        return is_distribution and not is_lying

    def is_cont_legal(self) -> bool:
        for y in self.game.messages:
            _sum = 0
            for x in self.game.outcomes:
                _sum += self.game.cont[y][x]

            if _sum != 1:
                return False

        return True

    def to_quiz_strategy(self, s: pu.PreStrategy) -> YgivenX:
        new_dict = {}
        for x in s.strategy.keys():
            new_dict[self.game.outcomes[x]] = {}
            for y in s.strategy[x].keys():
                new_dict[self.game.outcomes[x]][self.game.messages[y]] = s.strategy[x][y]

        return new_dict

    def to_cont_strategy(self, s: pu.PreStrategy) -> XgivenY:
        new_dict = {}
        for y in s.strategy.keys():
            new_dict[self.game.messages[y]] = {}
            for x in s.strategy[y].keys():
                new_dict[self.game.messages[y]][self.game.outcomes[x]] = s.strategy[y][x]

        return new_dict

    def to_pre_quiz_strategy(self, s: np.ndarray) -> pu.PreStrategy:
        return pu.PreStrategy("dynamic_quiz", {
            0: {
                0: 1,
                1: 0,
            },
            1: {
                0: s[0],
                1: 1 - s[0]
            },
            2: {
                0: 0,
                1: 1
            }
        })

    def to_pre_cont_strategy(self, s: np.ndarray) -> pu.PreStrategy:
        return pu.PreStrategy("dynamic_cont", {
            0: {
                0: s[0],
                1: 1 - s[0],
                2: 0
            },
            1: {
                0: 0,
                1: s[1],
                2: 1 - s[1],
            }
        })

    def is_car(self) -> bool:
        for y in self.game.messages:
            val = float('nan')
            for x in y.outcomes:
                if math.isnan(val):
                    val = self.game.quiz[x][y]
                elif val is not self.game.quiz[x][y]:
                    return False

        return True

    def is_rcar(self) -> bool:
        rcar_vector = [float('nan') for _ in range(len(self.game.outcomes))]
        for x in self.game.outcomes:
            for y in x.messages:
                test = self.game.quiz_reverse[y][x]
                if math.isnan(rcar_vector[x.id]):
                    rcar_vector[x.id] = self.game.quiz_reverse[y][x]
                elif rcar_vector[x.id] != self.game.quiz_reverse[y][x]:
                    return False

        return True
