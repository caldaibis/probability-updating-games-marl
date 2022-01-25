from __future__ import annotations

import random
import numpy as np
from typing import Dict

import src.probability_updating as pu
import src.probability_updating.games as games


class SimulationWrapper:
    game = games.Game

    def __init__(self, game: games.Game, actions: Dict[pu.Agent, np.ndarray]):
        self.game = game
        for agent in pu.AGENTS:
            self.game.set_action(agent, actions[agent])

    def _simulate_single(self) -> (pu.Outcome, pu.Message, Dict[pu.Agent, float], Dict[pu.Agent, float]):
        x = random.choices(list(self.game.marginal_outcome.keys()), list(self.game.marginal_outcome.values()), k=1)[0]
        vv = [self.game.action[pu.HOST][x, y] for y in x.messages]
        y = random.choices(x.messages, vv, k=1)[0]

        loss = {agent: self.game.get_loss(agent, x, y) for agent in pu.AGENTS}
        entropy = {agent: self.game.get_entropy(agent, y) for agent in pu.AGENTS}

        return x, y, loss, entropy

    def simulate(self, n: int) -> (Dict[pu.Outcome, int], Dict[pu.Message, int], Dict[pu.Agent, float], Dict[pu.Agent, float]):
        x_count = {x: 0 for x in self.game.outcomes}
        y_count = {y: 0 for y in self.game.messages}
        losses = {agent: [] for agent in pu.AGENTS}
        entropies = {agent: [] for agent in pu.AGENTS}

        for _ in range(n):
            x, y, loss, entropy = self._simulate_single()
            x_count[x] += 1
            y_count[y] += 1
            losses[pu.CONT].append(loss[pu.CONT])
            losses[pu.HOST].append(loss[pu.HOST])
            entropies[pu.CONT].append(entropy[pu.CONT])
            entropies[pu.HOST].append(entropy[pu.HOST])

        return x_count, y_count, \
               {agent: np.mean(losses[agent]) for agent in pu.AGENTS}, \
               {agent: np.mean(entropies[agent]) for agent in pu.AGENTS}
