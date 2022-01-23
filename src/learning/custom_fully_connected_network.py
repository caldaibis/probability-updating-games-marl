from __future__ import annotations

from typing import Dict, List

import gym
import numpy as np
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class CustomFullyConnectedNetwork(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Start Actor network -------------------------------------------------
        hiddens = list(model_config.get("fcnet_hiddens", [])) + \
                  list(model_config.get("post_fcnet_hiddens", [])) # no post hiddens
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation") # wont happen
        no_final_linear = model_config.get("no_final_linear") # False
        self.vf_share_layers = model_config.get("vf_share_layers") # False
        self.free_log_std = model_config.get("free_log_std") # False

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to last hidden layer.
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size
        self._hidden_layers = nn.Sequential(*layers)

        # Add a last linear layer of size num_outputs.
        self._logits = SlimFC(
            in_size=prev_layer_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        # End Actor network ---------------------------------------------------

        # Start Critic network ------------------------------------------------
        self._value_branch_separate = None
        # Build a parallel set of hidden layers for the value net.
        prev_vf_layer_size = int(np.product(obs_space.shape))
        vf_layers = []
        for size in hiddens:
            vf_layers.append(
                SlimFC(
                    in_size=prev_vf_layer_size,
                    out_size=size,
                    activation_fn=activation,
                    initializer=normc_initializer(1.0)))
            prev_vf_layer_size = size
        self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        # End Critic network --------------------------------------------------

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features)
        return logits, state

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"

        return self._value_branch(self._value_branch_separate(self._last_flat_in)).squeeze(1)

    def import_from_h5(self, h5_file: str) -> None:
        pass
