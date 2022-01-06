from __future__ import annotations

import random

from hyperopt import hp
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import sample_from
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
            "search_alg": HyperOptSearch
            (
                {
                    "train_batch_size": hp.randint("train_batch_size", 1000, 10000),
                    "sgd_minibatch_size": hp.randint("sgd_minibatch_size", 16, 256),
                    "num_sgd_iter": hp.randint("num_sgd_iter", 5, 30),
                    "lambda": hp.uniform("lambda", 0.9, 1.0),
                    "clip_param": hp.uniform("clip_param", 0.1, 0.5),
                    "lr": hp.uniform("lr", 1e-5, 1e-3)
                },
                metric="episode_reward_mean", mode="max",  # points_to_evaluate=
                # [{
                #     "train_batch_size": 1000,
                #     "sgd_minibatch_size": 16,
                #     "num_sgd_iter": 5,
                #     "lambda": 0.9,
                #     "clip_param": 0.1,
                #     "lr": 1e-3,
                # }]
            ),
        }

    def _create_model_config(self) -> dict:
        return {
            **super()._create_model_config(),
            "multiagent": {
                "policies": {"default_policy"},
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
            },
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.pad_action_space_v0(env)
        env = ss.agent_indicator_v0(env)

        return ParallelPettingZooEnv(env)
