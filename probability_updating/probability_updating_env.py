from __future__ import annotations

from typing import List, Dict, Union

import numpy as np
from gym import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils.agent_selector import agent_selector

import probability_updating as pu


def env(game: pu.Game):
    return ProbabilityUpdatingEnv(game)


class ProbabilityUpdatingEnv(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "probability_updating_game"}
    game: pu.Game
    observations: Dict[str, Union[List[List[bool]], List[float]]]

    def __init__(self, g: pu.Game):
        super().__init__()

        self.game = g

        self.seed()

        self.agents = pu.agents()
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        # Monty Hall
        # y1 < {x1, x2}
        # y2 < {x2, x3}
        # --
        # x1 < {y1}
        # x2 < {y1, y2}
        # x3 < {y2}

        # Colly Hall
        # y1 < {x1, x2}
        # y2 < {x2, x3}
        # y3 < {x2}
        # --
        # x1 < {y1}
        # x2 < {y1, y2, y3}
        # x3 < {y2}

        self.action_spaces = {
            pu.cont(): spaces.Box(low=0.0, high=1.0, shape=(sum(len(y.outcomes) - 1 for y in g.messages),), dtype=np.float32),
            pu.quiz(): spaces.Box(low=0.0, high=1.0, shape=(sum(len(x.messages) - 1 for x in g.outcomes),), dtype=np.float32)

        }

        self.raw_observation_space = spaces.Dict({
            'structure': spaces.Box(low=False, high=True, shape=(len(g.messages), len(g.outcomes)), dtype=bool),
            'marginal': spaces.Box(low=0.0, high=1.0, shape=(len(g.outcomes)-1,), dtype=np.float32)
        })

        self.observation_spaces = {agent: spaces.flatten_space(self.raw_observation_space) for agent in self.agents}

        self.observations = {
            'structure': [[x in y.outcomes for x in g.outcomes] for y in g.messages],
            'marginal': [g.marginal_outcome[x] for x in g.outcomes[0:-1]]
        }

        self.steps = 0
        self.rewards = None

    def seed(self, seed=None):
        """
        Reseeds the environment (making the resulting environment deterministic).
        `reset()` must be called after `seed()`, and before `step()`.
        """
        super().seed(seed)

    def close(self):
        """
        Closes the rendering window, subprocesses, network connections, or any other resources
        that should be released.
        """
        super().close()

    def observation_space(self, agent):
        """
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return self.action_spaces[agent]

    def step(self, actions):
        """
        Receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, done dictionary, and info dictionary,
        where each dictionary is keyed by the agent.
        """
        observations = {agent: spaces.flatten(self.raw_observation_space, self.observations) for agent in self.agents}
        losses = self.game.play(actions)
        dones = {agent: True for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.agents = []

        return observations, {a: -loss for a, loss in losses.items()}, dones, infos

    def render(self, mode="human"):
        """
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside
        of classic, and `'ansi'` which returns the strings printed
        (specific to classic environments).
        """
        print(f"Rendering... players: {pu.quiz()}, {pu.cont()}")

    def state(self):
        """
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        pass

    def reset(self):
        """
        resets the environment and returns a dictionary of observations (keyed by the agent name)
        """
        self.game.reset()

        self.agents = self.possible_agents[:]

        return {agent: spaces.flatten(self.raw_observation_space, self.observations) for agent in self.agents}

