from __future__ import annotations

from typing import Tuple

from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

import src.probability_updating as pu


class ProbabilityUpdatingEnvWrapper(MultiAgentEnv):
    env: pu.ProbabilityUpdatingEnv

    def __init__(self, env: pu.ProbabilityUpdatingEnv):
        self.env = env

        self.agents = self.env.possible_agents
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces

        self.reset()

    def reset(self) -> MultiAgentDict:
        return self.env.reset()

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obss, rews, dones, infos = self.env.step(action_dict)
        dones["__all__"] = all(dones.values())
        return obss, rews, dones, infos
