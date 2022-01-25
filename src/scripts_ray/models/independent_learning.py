from __future__ import annotations

from typing import Dict, Type
from ray.rllib import MultiAgentEnv
from ray.rllib.agents import Trainer
from ray.rllib.policy.policy import PolicySpec

import probability_updating as pu
from src.scripts_ray import Model, CustomMetricCallbacks, RayProbabilityUpdatingEnv

import supersuit as ss


class IndependentLearning(Model):
    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainer], hyper_param: Dict, min_total_time_s: int, max_total_time_s: int):
        super(IndependentLearning, self).__init__(game, losses, trainer_type, hyper_param, min_total_time_s, max_total_time_s)

        self.reporter.add_metric_column("universal_reward_mean")
        self.reporter.add_metric_column("universal_reward_eval_mean")
        
        self.reporter.add_metric_column("rcar_dist_mean")
        self.reporter.add_metric_column("rcar_dist_eval_mean")
        
        self.reporter.add_metric_column("reward_cont_mean")
        self.reporter.add_metric_column("reward_cont_eval_mean")
        
        self.reporter.add_metric_column("reward_host_mean")
        self.reporter.add_metric_column("reward_host_eval_mean")
        
        self.metric = "universal_reward_mean"
    
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
                "policies": {
                    pu.Agent.Cont.value: PolicySpec(None, self.env.observation_spaces[pu.Agent.Cont.value], self.env.action_spaces[pu.Agent.Cont.value], None),
                    pu.Agent.Host.value: PolicySpec(None, self.env.observation_spaces[pu.Agent.Host.value], self.env.action_spaces[pu.Agent.Host.value], None),
                },
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
            },
            "callbacks": CustomMetricCallbacks,
            "num_workers": 6,
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> MultiAgentEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.agent_indicator_v0(env)

        return RayProbabilityUpdatingEnv(env)

    def predict(self, checkpoint: str):
        trainer = self.trainer_type(config=self._create_model_config())
        trainer.restore(checkpoint)

        obs = self.env.reset()
        actions = {agent.value: trainer.compute_single_action(obs[agent.value], unsquash_action=True, explore=False, policy_id=agent.value) for agent in pu.Agent}
        obs, rewards, dones, infos = self.env.step(actions)
        print(self.game)
