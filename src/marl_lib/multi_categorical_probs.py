from __future__ import annotations

import gym
import numpy as np
from ray.rllib.models import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class MultiCategoricalProbs(TorchDistributionWrapper):
    """MultiCategorical distribution that produces actions for probability distribution action spaces."""
    def __init__(self,
                 inputs: List[TensorType],
                 model: TorchModelV2,
                 input_lens: Union[List[int], np.ndarray, Tuple[int, ...]]):
        super().__init__(inputs, model)
        # If input_lens is np.ndarray or list, force-make it a tuple.
        inputs_split = self.inputs.split(tuple(input_lens), dim=1)
        self.cats = [
            torch.distributions.categorical.Categorical(logits=input_)
            for input_ in inputs_split
        ]
        self.output_shape = model.model_config['custom_action_dist_shape']
        # Used in case we are dealing with an Int Box.
        # self.action_space = action_space

    def sample(self) -> TensorType:
        return torch.stack(self.cat.probs, dim=1)

    def deterministic_sample(self) -> TensorType:
        return torch.stack(self.cat.probs, dim=1)

    def logp(self, actions: TensorType) -> TensorType:
        # # If tensor is provided, unstack it into list.
        if isinstance(actions, torch.Tensor):
            # if isinstance(self.action_space, gym.spaces.Box):
            #     actions = torch.reshape(
            #         actions, [-1, int(np.product(self.action_space.shape))])
            actions = torch.unbind(actions, dim=1)
        logps = torch.stack([cat.log_prob(act) for cat, act in zip(self.cats, actions)])
        return torch.sum(logps, dim=0)

    def multi_entropy(self) -> TensorType:
        return torch.stack([cat.entropy() for cat in self.cats], dim=1)

    def entropy(self) -> TensorType:
        return torch.sum(self.multi_entropy(), dim=1)

    def multi_kl(self, other: ActionDistribution) -> TensorType:
        return torch.stack(
            [
                torch.distributions.kl.kl_divergence(cat, oth_cat)
                for cat, oth_cat in zip(self.cats, other.cats)
            ],
            dim=1,
        )

    def kl(self, other: ActionDistribution) -> TensorType:
        return torch.sum(self.multi_kl(other), dim=1)

    @staticmethod
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        # todo: check of dit klopt voor mijn action space!
        return np.sum(action_space.nvec)
