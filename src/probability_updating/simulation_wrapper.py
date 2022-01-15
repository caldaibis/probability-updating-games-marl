from __future__ import annotations

import random
import numpy as np
from typing import Dict

import probability_updating as pu
import probability_updating.games as games


class SimulationWrapper:
    game = games.Game

    def __init__(self, game: games.Game, actions: Dict[pu.Agent, np.ndarray]):
        self.game = game
        for agent in pu.Agent:
            self.game.set_action(agent, actions[agent])

    def _simulate_single(self) -> (pu.Outcome, pu.Message, Dict[pu.Agent, float], Dict[pu.Agent, float]):
        x = random.choices(list(self.game.marginal_outcome.keys()), list(self.game.marginal_outcome.values()), k=1)[0]
        vv = [self.game.action[pu.Agent.Host][x, y] for y in x.messages]
        y = random.choices(x.messages, vv, k=1)[0]

        loss = {agent: self.game.get_loss(agent, x, y) for agent in pu.Agent}
        entropy = {agent: self.game.get_entropy(agent, y) for agent in pu.Agent}

        return x, y, loss, entropy

    def simulate(self, n: int) -> (Dict[pu.Outcome, int], Dict[pu.Message, int], Dict[pu.Agent, float], Dict[pu.Agent, float]):
        x_count = {x: 0 for x in self.game.outcomes}
        y_count = {y: 0 for y in self.game.messages}
        losses = {agent: [] for agent in pu.Agent}
        entropies = {agent: [] for agent in pu.Agent}

        for _ in range(n):
            x, y, loss, entropy = self._simulate_single()
            x_count[x] += 1
            y_count[y] += 1
            losses[pu.Agent.Cont].append(loss[pu.Agent.Cont])
            losses[pu.Agent.Host].append(loss[pu.Agent.Host])
            entropies[pu.Agent.Cont].append(entropy[pu.Agent.Cont])
            entropies[pu.Agent.Host].append(entropy[pu.Agent.Host])

        return x_count, y_count, \
               {agent: np.mean(losses[agent]) for agent in pu.Agent}, \
               {agent: np.mean(entropies[agent]) for agent in pu.Agent}
