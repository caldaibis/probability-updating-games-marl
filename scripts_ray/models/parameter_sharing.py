from __future__ import annotations

import random

from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import sample_from
from ray.tune.schedulers.pb2 import PB2

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
            # "callbacks": [scripts_ray.CustomCallback()],
            "scheduler": PB2
            (
                perturbation_interval=2.0,
                # Specifies the hyperparam search space
                hyperparam_bounds={
                    "train_batch_size": [1000, 10000],
                    "sgd_minibatch_size": [16, 256],
                    "num_sgd_iter": [5, 30],
                    "lambda": [0.9, 1.0],
                    "clip_param": [0.1, 0.5],
                    "lr": [1e-3, 1e-5],
                }
            ),
        }

    def _create_model_config(self) -> dict:
        return {
            **super()._create_model_config(),
            "multiagent": {
                "policies": {"default_policy"},
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
            },
            "train_batch_size": sample_from(lambda spec: random.randint(1000, 10000)),
            "sgd_minibatch_size": sample_from(lambda spec: random.randint(16, 256)),
            "num_sgd_iter": sample_from(lambda spec: random.randint(5, 30)),
            "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
            "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
            "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.pad_action_space_v0(env)
        env = ss.agent_indicator_v0(env)

        return ParallelPettingZooEnv(env)
