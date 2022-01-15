from __future__ import annotations

from typing import Dict, Type

from ray.rllib.agents import Trainer
from ray.rllib.env import ParallelPettingZooEnv

from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.pb2 import PB2
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

import probability_updating as pu
from src.scripts_ray import Model

import supersuit as ss


class ParameterSharingModel(Model):
    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainer], hyper_param: Dict, min_total_time_s: int, max_total_time_s: int):
        super(ParameterSharingModel, self).__init__(game, losses, trainer_type, hyper_param, min_total_time_s, max_total_time_s)

        self.metric = "episode_reward_mean"

    def get_local_dir(self) -> str:
        return f"output_ray/parameter_sharing/{self.trainer_type.__name__}/"

    def _create_tune_config(self) -> dict:
        return {
            **super()._create_tune_config(),
            "num_samples": 1,
            # "scheduler": PB2
            # (
            #     perturbation_interval=30.0,
            #     # Specifies the hyperparam search space
            #     hyperparam_bounds={
            #         "train_batch_size": [64, 128],
            #         "sgd_minibatch_size": [4, 16],
            #     }
            # ),
        }

    def _create_model_config(self) -> dict:
        return {
            **super()._create_model_config(),
            "multiagent": {
                "policies": {"default_policy"},
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
            },
            "num_workers": 6,
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.pad_action_space_v0(env)
        env = ss.agent_indicator_v0(env)

        return ParallelPettingZooEnv(env)

    def predict(self, checkpoint: str):
        model = self.trainer_type(config=self._create_model_config())
        model.restore(checkpoint)

        obs = self.env.reset()
        actions = {agent.value: model.compute_single_action(obs[agent.value], unsquash_action=True, explore=False) for agent in pu.Agent}

        obs, rewards, dones, infos = self.env.step(actions)
