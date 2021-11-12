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
            # confirm sums to one
            if not math.isclose(_sum, 1, rel_tol=1e-5):
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
            # confirm sums to one
            if not math.isclose(_sum, 1, rel_tol=1e-5):
                return False

        return True

    def to_cont_strategy(self, s: np.ndarray) -> pu.XgivenY:
        i = 0
        strategy = {y: {x: 0 for x in self.game.outcomes} for y in self.game.messages}

        for y in self.game.messages:
            sum_prob = 0

            if len(y.outcomes) == 1:
                x = y.outcomes[0]
                strategy[y][x] = 1
                continue

            for x in y.outcomes:
                # arrange values from gym env action space to original strategy space
                strategy[y][x] = s[i]
                sum_prob += s[i]
                i += 1

            # normalise to create a probability distribution: sum must be one
            if sum_prob == 0:
                for x in y.outcomes:
                    strategy[y][x] = 1 / len(y.outcomes)
                continue

            for x in y.outcomes:
                strategy[y][x] /= sum_prob

        return strategy

    def to_quiz_strategy(self, s: np.ndarray) -> pu.YgivenX:
        i = 0
        strategy = {x: {y: 0 for y in self.game.messages} for x in self.game.outcomes}

        for x in self.game.outcomes:
            sum_prob = 0

            if len(x.messages) == 1:
                y = x.messages[0]
                strategy[x][y] = 1
                continue

            for y in x.messages:
                # arrange values from gym env action space to original strategy space
                strategy[x][y] = s[i]
                sum_prob += s[i]
                i += 1

            # normalise to create a probability distribution: sum must be one
            if sum_prob == 0:
                for y in x.messages:
                    strategy[x][y] = 1 / len(x.messages)
                continue

            for y in x.messages:
                strategy[x][y] /= sum_prob

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
