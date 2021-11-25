from __future__ import annotations

import math
import random
import numpy as np
from typing import Dict

import probability_updating as pu
import probability_updating.games as games


class SimulationWrapper:
    game = games.Game

    def __init__(self, game: games.Game, actions: Dict[pu.Agent, np.ndarray]):
        self.game = game
        self.game.cont = actions[pu.cont()]
        self.game.quiz = actions[pu.quiz()]

    def _simulate_single(self) -> (pu.Outcome, pu.Message, Dict[pu.Agent, float], Dict[pu.Agent, float]):
        x = random.choices(list(self.game.marginal_outcome.keys()), list(self.game.marginal_outcome.values()), k=1)[0]
        y = random.choices(list(self.game.quiz[x].keys()), list(self.game.quiz[x].values()), k=1)[0]

        loss = {agent: self.game.get_loss(agent, x, y) for agent in pu.agents()}
        entropy = {agent: self.game.get_entropy(agent, y) for agent in pu.agents()}

        return x, y, loss, entropy

    def simulate(self, n: int) -> (Dict[pu.Outcome, int], Dict[pu.Message, int], Dict[pu.Agent, float], Dict[pu.Agent, float]):
        x_count = {x: 0 for x in self.game.outcomes}
        y_count = {y: 0 for y in self.game.messages}
        losses = {agent: [] for agent in pu.agents()}
        entropies = {agent: [] for agent in pu.agents()}

        for _ in range(n):
            x, y, loss, entropy = self._simulate_single()
            x_count[x] += 1
            y_count[y] += 1
            losses[pu.cont()].append(loss[pu.cont()])
            losses[pu.quiz()].append(loss[pu.quiz()])
            entropies[pu.cont()].append(entropy[pu.cont()])
            entropies[pu.quiz()].append(entropy[pu.quiz()])

        return x_count, y_count, \
               {agent: np.mean(losses[agent]) for agent in pu.agents()}, \
               {agent: np.mean(entropies[agent]) for agent in pu.agents()}
