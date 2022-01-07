from __future__ import annotations

import random
from typing import Dict

from hyperopt import hp
from ray import tune
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.pb2 import PB2
from ray.tune.suggest.hyperopt import HyperOptSearch

import probability_updating as pu
import scripts_ray
from scripts_ray import Model

import supersuit as ss


class ParameterSharingModel(Model):
    def get_local_dir(self) -> str:
        return f"output_ray/parameter_sharing/{self.trainer_type.__name__}/"

    def _create_tune_config(self, timeout_seconds: int, hyper_param: Dict) -> dict:
        return {
            **super()._create_tune_config(timeout_seconds, hyper_param),
            # "callbacks": [scripts_ray.CustomCallback()],
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
