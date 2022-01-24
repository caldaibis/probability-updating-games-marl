from __future__ import annotations

import math
from typing import Dict
import src.probability_updating as pu
import src.probability_updating.games as games


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
                    reverse[y][x] = self.game.action[pu.Agent.Host][x, y] * self.game.marginal_outcome[x] / self.game.marginal_message[y]

        return pu.ContAction(reverse)

    def update_message_marginal(self) -> Dict[pu.Message, float]:
        probs = {}
        for y in self.game.messages:
            probs[y] = 0
            for x in self.game.outcomes:
                probs[y] += self.game.action[pu.Agent.Host][x, y] * self.game.marginal_outcome[x]

        return probs

    def is_car(self) -> bool:
        for y in self.game.messages:
            val = float('nan')
            for x in y.outcomes:
                if math.isnan(val):
                    val = self.game.action[pu.Agent.Host][x, y]
                elif not math.isclose(val, self.game.action[pu.Agent.Host][x, y], rel_tol=1e-5):
                    return False

        return True

    def is_rcar(self) -> bool:
        rcar_vector = [float('nan') for _ in range(len(self.game.outcomes))]
        for x in self.game.outcomes:
            for y in x.messages:
                if math.isnan(rcar_vector[x.id]):
                    rcar_vector[x.id] = self.game.host_reverse[x, y]
                elif not math.isclose(rcar_vector[x.id], self.game.host_reverse[x, y], rel_tol=1e-5):
                    return False

        return True

    """Returns the SSE (sum of squared errors) or RSS (residual sum of squares) between the actual values and the RCAR values"""
    def rcar_sse(self) -> float:
        _sum = 0
        rcar_vector = [float('nan') for _ in range(len(self.game.outcomes))]
        for x in self.game.outcomes:
            for y in x.messages:
                if math.isnan(rcar_vector[x.id]):
                    rcar_vector[x.id] = self.game.host_reverse[x, y]
                _sum += math.pow(rcar_vector[x.id] - self.game.host_reverse[x, y], 2)

        return _sum

    """Returns the MSE (mean square error) between the actual values and the RCAR values"""
    def rcar_mse(self) -> float:
        _sum = 0
        _n = 0
        rcar_vector = [float('nan') for _ in range(len(self.game.outcomes))]
        for x in self.game.outcomes:
            for y in x.messages:
                if math.isnan(rcar_vector[x.id]):
                    rcar_vector[x.id] = self.game.host_reverse[x, y]
                _sum += math.pow(rcar_vector[x.id] - self.game.host_reverse[x, y], 2)
                _n += 1

        return _sum / _n

    """Returns the RMSE (root mean square error) between the actual values and the RCAR values"""
    def rcar_rmse(self) -> float:
        return math.sqrt(self.rcar_mse())
