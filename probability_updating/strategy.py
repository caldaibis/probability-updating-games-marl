from __future__ import annotations

import numpy as np
import math
from typing import Dict
import probability_updating as pu
import probability_updating.games as games


class Strategy:
    game: games.Game

    def __init__(self, game: games.Game):
        self.game = game

    def update_strategy_quiz_reverse(self) -> pu.XgivenY:
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

    def is_cont_legal(self) -> bool:
        for y in self.game.messages:
            _sum = 0
            for x in self.game.outcomes:
                val = self.game.cont[y][x]
                # confirm within domain [0, 1]
                if val < 0 or val > 1:
                    return False
                _sum += val
            # confirm sums to one (or zero, which we will allow but the algorithm will be punished with a high loss)
            if not math.isclose(_sum, 1, rel_tol=1e-5) and not math.isclose(_sum, 0, rel_tol=1e-5):
                return False

        return True

    def is_quiz_legal(self) -> bool:
        for x in self.game.outcomes:
            _sum = 0
            for y in self.game.messages:
                val = self.game.quiz[x][y]
                # confirm within domain [0, 1]
                if val < 0 or val > 1:
                    return False
                # confirm not lying (no positive values for outcomes not contained in message
                if val > 0 and x not in y.outcomes:
                    return False
                _sum += val
            # confirm sums to one (or zero, which we will allow but the algorithm will be punished with a high loss)
            if not math.isclose(_sum, 1, rel_tol=1e-5) and not math.isclose(_sum, 0, rel_tol=1e-5):
                return False

        return True

    def is_cont_all_zeroes(self):
        for y in self.game.messages:
            if sum(self.game.cont[y][x] for x in y.outcomes) == 0:
                return True

        return False

    def is_quiz_all_zeroes(self):
        for x in self.game.outcomes:
            if sum(self.game.quiz[x][y] for y in x.messages) == 0:
                return True

        return False

    def to_cont_strategy(self, s: np.ndarray) -> pu.XgivenY:
        i = 0
        strategy = {y: {x: 0 for x in self.game.outcomes} for y in self.game.messages}

        for y in self.game.messages:
            sum_prob = 0
            for x in y.outcomes:
                if x == y.outcomes[-1]:
                    strategy[y][x] = 1 - sum_prob
                else:
                    strategy[y][x] = s[i]
                    sum_prob += s[i]
                    i += 1

        return strategy

    def to_quiz_strategy(self, s: np.ndarray) -> pu.YgivenX:
        i = 0
        strategy = {x: {y: 0 for y in self.game.messages} for x in self.game.outcomes}

        for x in self.game.outcomes:
            sum_prob = 0
            for y in x.messages:
                if y == x.messages[-1]:
                    strategy[x][y] = 1 - sum_prob
                else:
                    strategy[x][y] = s[i]
                    sum_prob += s[i]
                    i += 1

        return strategy

    def is_car(self) -> bool:
        for y in self.game.messages:
            val = float('nan')
            for x in y.outcomes:
                if math.isnan(val):
                    val = self.game.quiz[x][y]
                elif not math.isclose(val, self.game.quiz[x][y], rel_tol=1e-5):
                    return False

        return True

    def is_rcar(self) -> bool:
        rcar_vector = [float('nan') for _ in range(len(self.game.outcomes))]
        for x in self.game.outcomes:
            for y in x.messages:
                if math.isnan(rcar_vector[x.id]):
                    rcar_vector[x.id] = self.game.quiz_reverse[y][x]
                elif not math.isclose(rcar_vector[x.id], self.game.quiz_reverse[y][x], rel_tol=1e-5):
                    return False

        return True

