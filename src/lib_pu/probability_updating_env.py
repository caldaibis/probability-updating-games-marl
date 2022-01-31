from __future__ import annotations

from typing import Dict

from gym import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils.agent_selector import agent_selector

import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_marl as marl


class ProbabilityUpdatingEnv(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "probability_updating_game"}

    game: pu_games.Game

    action_spaces: Dict[str, spaces.Tuple]
    observation_spaces: Dict[str, spaces.Box]

    def __init__(self, g: pu_games.Game):
        super().__init__()

        self.game = g

        self.seed()

        self.agents = [agent for agent in pu.AGENTS]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {
            agent: spaces.Tuple([marl.CustomSimplex(actions) for actions in g.get_action_shape(agent)])
            for agent in pu.AGENTS
        }
        self.observation_spaces = {
            agent: spaces.Discrete(1) for agent in pu.AGENTS
        }

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
        losses = self.game.step(actions)

        observations = {agent: 0 for agent in self.agents}
        rewards = {agent: -loss for agent, loss in losses.items()}
        dones = {agent: True for agent in self.agents}
        infos = {
            pu.CONT: {"expected_entropy": self.game.get_expected_entropy()},
            pu.HOST: {"rcar_dist": self.game.strategy_util.rcar_dist()},
        }

        return observations, rewards, dones, infos

    def render(self, mode="human"):
        """
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside
        of classic, and `'ansi'` which returns the strings printed
        (specific to classic environments).
        """
        pass

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
        self.agents = self.possible_agents[:]

        return {agent: 0 for agent in self.agents}
