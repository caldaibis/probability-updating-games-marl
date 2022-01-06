from __future__ import annotations

import random

from hyperopt import hp
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import sample_from
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import probability_updating as pu
import scripts_ray
from scripts_ray import Model

import supersuit as ss


class ParameterSharingModel(Model):
    def get_local_dir(self) -> str:
        return f"output_ray/parameter_sharing/{self.trainer_type.__name__}"

    def _create_tune_config(self, timeout_seconds: int) -> dict:
        return {
            **super()._create_tune_config(timeout_seconds),
            "callbacks": [scripts_ray.CustomCallback()],
            "num_samples": 4,
            "scheduler": ASHAScheduler
            (
                time_attr='training_iteration',
                max_t=10,
                grace_period=5,
                reduction_factor=4,
            ),
        }

    def _create_model_config(self) -> dict:
        return {
            **super()._create_model_config(),
            "multiagent": {
                "policies": {"default_policy"},
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
            },
            "num_workers": 1,
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.pad_action_space_v0(env)
        env = ss.agent_indicator_v0(env)

        return ParallelPettingZooEnv(env)
