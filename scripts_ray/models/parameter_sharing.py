from __future__ import annotations

from typing import Type, Dict, Optional

from ray.rllib.agents import Trainer
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.tune import Trainable

import probability_updating as pu
import scripts_ray
from scripts_ray import Model

import supersuit as ss


class ParameterSharingModel(Model):

    @classmethod
    def get_local_dir(cls) -> str:
        return "output_ray_ps"

    @classmethod
    def _create_tune_config(cls, iterations: int) -> dict:
        return {
            **super(cls)._create_tune_config(iterations),
            "callbacks": [scripts_ray.CustomCallback()],
        }

    @classmethod
    def _create_model_config(cls) -> dict:
        return {
            **super()._create_model_config(),
            "multiagent": {
                "policies": {"default_policy"},
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
            },
            # "custom_eval_function": custom_eval_function,
            # "model": {
            #     "vf_share_layers": False,
            # },
            # "vf_loss_coeff": 0.01,
            # "train_batch_size": 10,
            # "sgd_minibatch_size": 1,
            # "num_sgd_iter": 30,
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.pad_action_space_v0(env)
        env = ss.agent_indicator_v0(env)

        return ParallelPettingZooEnv(env)

    def custom_eval_function(self, trainer: Trainer, eval_workers: WorkerSet):
        pass
