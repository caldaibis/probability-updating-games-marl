from __future__ import annotations

import math
from typing import Dict
import src.probability_updating as pu
import src.probability_updating.games as games
import statistics


class StrategyUtil:
    game: games.Game

    def __init__(self, game: games.Game):
        self.game = game

    def update_strategy_host_reverse(self) -> pu.ContAction:
        reverse = {}
        for y in self.game.messages:
            reverse[y] = {}
            for x in self.game.outcomes:
                if self.game.marginal_message[y] == 0:
                    reverse[y][x] = 0
                else:
                    reverse[y][x] = self.game.action[pu.HOST][x, y] * self.game.marginal_outcome[x] / self.game.marginal_message[y]

        return pu.ContAction(reverse)

    def update_message_marginal(self) -> Dict[pu.Message, float]:
        probs = {}
        for y in self.game.messages:
            probs[y] = 0
            for x in self.game.outcomes:
                probs[y] += self.game.action[pu.HOST][x, y] * self.game.marginal_outcome[x]

        return probs

    def is_car(self) -> bool:
        for y in self.game.messages:
            val = float('nan')
            for x in y.outcomes:
                if math.isnan(val):
                    val = self.game.action[pu.HOST][x, y]
                elif not math.isclose(val, self.game.action[pu.HOST][x, y], rel_tol=1e-5):
                    return False

        return True

    def is_rcar(self) -> bool:
        rcar_vector = {x: float('nan') for x in self.game.outcomes}
        for x in self.game.outcomes:
            for y in x.messages:
                if math.isnan(rcar_vector[x]):
                    rcar_vector[x] = self.game.host_reverse[x, y]
                elif not math.isclose(rcar_vector[x], self.game.host_reverse[x, y], rel_tol=1e-5):
                    return False

        return True

    """Returns an approximation of the distance between a potential RCAR vector and the actual vector"""
    def rcar_dist(self) -> float:
        dist = 0
        for x in self.game.outcomes:
            mean = statistics.mean(self.game.host_reverse[x, y] for y in x.messages)
            dist += math.sqrt(sum(math.pow(mean - self.game.host_reverse[x, y], 2) for y in x.messages))

        return dist
