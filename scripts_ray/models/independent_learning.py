from __future__ import annotations

from typing import Dict
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec

import probability_updating as pu
import scripts_ray
from scripts_ray import Model, CustomMetricCallbacks

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
                # "policies": set(self.env.agents),
                "policies": {
                    pu.Agent.Cont.value: PolicySpec(None, self.env.observation_spaces[pu.Agent.Cont.value], self.env.action_spaces[pu.Agent.Cont.value], None),
                    pu.Agent.Host.value: PolicySpec(None, self.env.observation_spaces[pu.Agent.Host.value], self.env.action_spaces[pu.Agent.Host.value], None),
                },
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
            },
            "num_workers": 0,
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> MultiAgentEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.agent_indicator_v0(env)

        return scripts_ray.RayProbabilityUpdatingEnv(env)

    def predict(self, checkpoint: str):
        trainer = self.trainer_type(config=self._create_model_config())
        trainer.restore(checkpoint)

        obs = self.env.reset()
        actions = {agent.value: trainer.compute_single_action(obs[agent.value], unsquash_action=True, explore=False, policy_id=agent.value) for agent in pu.Agent}
        obs, rewards, dones, infos = self.env.step(actions)
