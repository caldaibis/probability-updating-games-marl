from __future__ import annotations

import math
import random
import numpy as np
from typing import Dict

import probability_updating as pu


class SimulationWrapper:
    game = pu.game.Game

    def __init__(self, game: pu.Game):
        self.game = game

    def simulate_single(self) -> (pu.Outcome, pu.Message, Dict[pu.Agent, float], Dict[pu.Agent, float]):
        x = random.choices(list(self.game.marginal_outcome.keys()), list(self.game.marginal_outcome.values()), k=1)[0]
        y = random.choices(list(self.game.quiz[x].keys()), list(self.game.quiz[x].values()), k=1)[0]

        loss = {agent: self.game.loss_fn[agent](self.game.cont, x, y) for agent in pu.agents()}
        entropy = {agent: self.game.entropy_fn[agent](y) if callable(self.game.entropy_fn[agent]) else math.nan for
                   agent in pu.agents()}

        return x, y, loss, entropy

    def simulate(self, n: int) -> (Dict[pu.Outcome, int], Dict[pu.Message, int], Dict[pu.Agent, float], Dict[pu.Agent, float]):
        x_count = {x: 0 for x in self.game.outcomes}
        y_count = {y: 0 for y in self.game.messages}
        losses = {agent: [] for agent in pu.agents()}
        entropies = {agent: [] for agent in pu.agents()}

        for _ in range(n):
            x, y, loss, entropy = self.simulate_single()
            x_count[x] += 1
            y_count[y] += 1
            losses[pu.cont()].append(loss[pu.cont()])
            losses[pu.quiz()].append(loss[pu.quiz()])
            entropies[pu.cont()].append(entropy[pu.cont()])
            entropies[pu.quiz()].append(entropy[pu.quiz()])

        return x_count, y_count, \
               {agent: np.mean(losses[agent]) for agent in pu.agents()}, \
               {agent: np.mean(entropies[agent]) for agent in pu.agents()}
