from __future__ import annotations

import os
from typing import Dict, Optional, Type

import gym
import gym.spaces
import numpy as np

import ray.tune
from ray.rllib import Policy, SampleBatch, RolloutWorker
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.examples.models.centralized_critic_models import YetAnotherTorchCentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.tune import Trainable

import probability_updating as pu
from scripts_ray import Model


class CentralisedCriticModel(Model):
    @classmethod
    def get_local_dir(cls) -> str:
        return "output_ray_cc"

    @classmethod
    def _create_tune_config(cls, iterations: int) -> dict:
        return {
            **super(cls)._create_tune_config(iterations),

        }

    observation_space: gym.spaces.Dict

    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], model_type: Type[Trainable], ext_name: Optional[str] = ''):
        super().__init__(game, losses, model_type, ext_name)

        self.observation_space = gym.spaces.Dict({
            "obs": self.env.observation_space,
            "opponent_action": self.env.action_space
        })

        ModelCatalog.register_custom_model("cc_model", YetAnotherTorchCentralizedCriticModel)

    def _create_model_config(self) -> dict:
        return {
            **super()._create_model_config(),
            "multiagent": {
                "policies": {
                    "cont": (None, self.observation_space, self.action_space, {}),
                    "quiz": (None, self.observation_space, self.action_space, {}),
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: "cont" if agent_id == 0 else "quiz",
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

