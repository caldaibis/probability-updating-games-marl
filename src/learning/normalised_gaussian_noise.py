from __future__ import annotations

from typing import Union
from ray.rllib.models import ActionDistribution
from ray.rllib.utils.exploration import GaussianNoise
from ray.rllib.utils.framework import TensorType


class NormalisedGaussianNoise(GaussianNoise):
    def _get_torch_exploration_action(self,
                                      action_dist: ActionDistribution,
                                      explore: bool,
                                      timestep: Union[int, TensorType]):
        action, logp = super()._get_torch_exploration_action(action_dist, explore, timestep)

        # Apply additional normalisation over actions to enforce a distribution
        # todo!
        print(action)

        return action, logp