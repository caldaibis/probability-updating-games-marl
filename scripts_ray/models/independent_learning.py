from __future__ import annotations

from typing import Dict
from ray import tune
from ray.rllib.env import ParallelPettingZooEnv

import probability_updating as pu
from scripts_ray import Model

import supersuit as ss


class IndependentLearning(Model):
    def get_local_dir(self) -> str:
        return f"output_ray/independent_learning/{self.trainer_type.__name__}/"

    def _create_tune_config(self) -> dict:
        return {
            **super()._create_tune_config(),
            "num_samples": 1,
        }

    def _create_model_config(self) -> dict:
        return {
            **super()._create_model_config(),
            "multiagent": {
                "policies": set(self.env.agents),
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
            },
            "num_workers": 0,
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.pad_action_space_v0(env)

        return ParallelPettingZooEnv(env)

    def predict(self, checkpoint: str):
        model = self.trainer_type(config=self._create_model_config())
        model.restore(checkpoint)

        obs = self.env.reset()
        actions = {agent.value: model.get_policy(agent.value).compute_single_action(obs[agent.value], explore=False)[0] for agent in pu.Agent}

        obs, rewards, dones, infos = self.env.step(actions)
