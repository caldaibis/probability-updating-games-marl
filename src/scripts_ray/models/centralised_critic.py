from __future__ import annotations

from typing import Dict, Optional

import gym
import gym.spaces
import numpy as np
import supersuit as ss

import tensorflow as tf
from gym.spaces import Box

from ray.rllib import Policy, SampleBatch, RolloutWorker
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf import TFModelV2, FullyConnectedNetwork
from ray.rllib.utils.typing import AgentID, PolicyID

import probability_updating as pu
from src.scripts_ray import Model


class CentralisedCriticModel(Model):
    observation_space: gym.spaces.Dict

    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], ext_name: Optional[str] = ''):
        super().__init__(game, losses, ext_name)

        self.observation_space = gym.spaces.Dict({
            "obs": self.env.observation_space,
            "opponent_action": self.env.action_space
        })

        ModelCatalog.register_custom_model("cc_model", self.CentralisedCriticModel)

    def get_local_dir(self) -> str:
        return "output_ray/centralised_critic"

    def _create_tune_config(self, timeout_seconds: int) -> dict:
        return {
            **super()._create_tune_config(timeout_seconds),
        }

    def _create_model_config(self) -> dict:
        return {
            **super()._create_model_config(),
            "multiagent": {
                "policies": {
                    "cont": (None, self.observation_space, self.env.action_space, {}),
                    "host": (None, self.observation_space, self.env.action_space, {}),
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: "cont" if agent_id == 0 else "host",
                "observation_fn": self.central_critic_observer
            },
            "model": {
                "custom_model": "cc_model"
            },
            "callbacks": self.FillInActions,
            # "batch_mode": "complete_episodes",
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.pad_action_space_v0(env)

        return ParallelPettingZooEnv(env)

    @staticmethod
    def central_critic_observer(agent_obs, **kw):
        """Rewrites the agent obs to include opponent data for training."""
        return {
            0: {
                "obs": agent_obs[0],
                "opponent_action": 0,  # filled in by FillInActions
            },
            1: {
                "obs": agent_obs[1],
                "opponent_action": 0,  # filled in by FillInActions
            },
        }

    class CentralisedCriticModel(TFModelV2):
        """Multi-agent model that implements a centralized value function.

        It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
        former of which can be used for computing actions (i.e., decentralized
        execution), and the latter for optimization (i.e., centralized learning).

        This model has two parts:
        - An action model that looks at just 'own_obs' to compute actions
        - A value model that also looks at the 'opponent_obs' / 'opponent_action'
          to compute the value (it does this by using the 'obs_flat' tensor).
        """

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            super().__init__(obs_space, action_space, num_outputs, model_config, name)

            self.action_model = FullyConnectedNetwork(
                Box(low=0, high=1, shape=(2,)),  # one-hot encoded Discrete(6)
                action_space,
                num_outputs,
                model_config,
                name + "_action")

            self.value_model = FullyConnectedNetwork(obs_space, action_space, 1, model_config, name + "_vf")

        def forward(self, input_dict, state, seq_lens):
            print(input_dict)
            self._value_out, _ = self.value_model({"obs": input_dict["obs_flat"]}, state, seq_lens)
            return self.action_model({"obs": input_dict["obs"]["obs"]}, state, seq_lens)

        def value_function(self):
            return tf.reshape(self._value_out, [-1])

    class FillInActions(DefaultCallbacks):
        """Fills in the opponent actions info in the training batches."""

        def on_postprocess_trajectory(self, *, worker: "RolloutWorker", episode: MultiAgentEpisode, agent_id: AgentID,
                                      policy_id: PolicyID, policies: Dict[PolicyID, Policy],
                                      postprocessed_batch: SampleBatch, original_batches: Dict[AgentID, SampleBatch],
                                      **kwargs) -> None:
            to_update = postprocessed_batch[SampleBatch.CUR_OBS]
            other_id = 1 if agent_id == 0 else 0
            action_encoder = ModelCatalog.get_preprocessor_for_space(policies[policy_id].action_space)

            # set the opponent actions into the observation
            _, opponent_batch = original_batches[other_id]
            opponent_actions = np.array([action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]])
            to_update[:, -2:] = opponent_actions
