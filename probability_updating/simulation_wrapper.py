from __future__ import annotations

import random
from statistics import mean
from typing import Dict

import probability_updating as pu


class SimulationWrapper:
    game = pu.game.Game

    def __init__(self, game: pu.Game):
        self.game = game

    def simulate_single(self) -> (pu.Outcome, pu.Message, float, float):
        x = random.choices(list(self.game.marginal_outcome.keys()), list(self.game.marginal_outcome.values()), k=1)[0]
        y = random.choices(list(self.game.quiz[x].keys()), list(self.game.quiz[x].values()), k=1)[0]
        loss = self.game.loss.get_loss(x, y)
        entropy = self.game.loss.get_entropy(y)

        return x, y, loss, entropy

    def simulate(self, n: int) -> (Dict[pu.Outcome, int], Dict[pu.Message, int], float, float):
        x_count = {x: 0 for x in self.game.outcomes}
        y_count = {y: 0 for y in self.game.messages}
        losses = []
        entropies = []

        for _ in range(n):
            x, y, loss, entropy = self.simulate_single()
            x_count[x] += 1
            y_count[y] += 1
            losses.append(loss)
            entropies.append(entropy)

        return x_count, y_count, mean(losses), mean(entropies)
