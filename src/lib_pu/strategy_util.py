from __future__ import annotations

import math
from typing import Dict
import src.lib_pu as pu
import src.lib_pu.games as games
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
                if self.game.message_dist[y] == 0:
                    reverse[y][x] = 0
                else:
                    reverse[y][x] = self.game.action[pu.HOST][x, y] * self.game.outcome_dist[x] / self.game.message_dist[y]

        return pu.ContAction(reverse)

    def update_message_dist(self) -> Dict[pu.Message, float]:
        probs = {}
        for y in self.game.messages:
            probs[y] = 0
            for x in self.game.outcomes:
                probs[y] += self.game.action[pu.HOST][x, y] * self.game.outcome_dist[x]

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
        return self.rcar_dist() < pu.RCAR_EPSILON
        # rcar_vec = {x: None for x in self.game.outcomes}
        # for x in self.game.outcomes:
        #     for y in [y for y in x.messages if self.game.message_dist[y] > 0]:
        #         if not rcar_vec[x]:
        #             rcar_vec[x] = self.game.host_reverse[x, y]
        #         elif not math.isclose(rcar_vec[x], self.game.host_reverse[x, y], rel_tol=1e-2):
        #             return False
        #
        # for y in self.game.messages:
        #     if sum(rcar_vec[x] for x in y.outcomes) > 1:
        #         return False
        #
        # return True

    """Returns an approximation of the distance between a potential RCAR vector and the actual vector"""
    def rcar_dist(self) -> float:
        dist = 0
        for x in self.game.outcomes:
            mean = statistics.mean(self.game.host_reverse[x, y] for y in x.messages if self.game.message_dist[y] > 0)
            dist += math.sqrt(sum((mean - self.game.host_reverse[x, y]) ** 2 for y in x.messages))
        
        for y in self.game.messages:
            dist += max(0, sum(self.game.host_reverse[x, y] for x in y.outcomes) - 1)
        
        return dist
