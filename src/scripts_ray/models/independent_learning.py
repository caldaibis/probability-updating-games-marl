from __future__ import annotations

from typing import Dict, Type
from ray.rllib import MultiAgentEnv
from ray.rllib.agents import Trainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

import probability_updating as pu
import src.scripts_ray as custom

import supersuit as ss


class IndependentLearning(custom.Model):
    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainer], hyper_param: Dict, min_total_time_s: int, max_total_time_s: int):
        super(IndependentLearning, self).__init__(game, losses, trainer_type, hyper_param, min_total_time_s, max_total_time_s)

        self.reporter.add_metric_column("surrogate_reward_mean")
        self.reporter.add_metric_column("policy_reward_mean_cont")
        self.reporter.add_metric_column("policy_reward_mean_host")
        self.reporter.add_metric_column("policy_reward_mean_min")
        self.reporter.add_metric_column("policy_reward_mean_max")

        self.metric = "surrogate_reward_mean"

        # dist_class, logit_dim = ModelCatalog.get_action_dist(
        #     self.env.action_space[pu.Agent.Cont],
        #     model_config_temp["model"],
        #     framework='torch')
        #
        # custom_fcnet = ModelCatalog.get_model_v2(
        #     obs_space=self.env.observation_space[pu.Agent.Cont],
        #     action_space=self.env.action_space[pu.Agent.Cont],
        #     num_outputs=2,
        #     model_config=MODEL_DEFAULTS,
        #     framework=args.framework,
        #     # Providing the `model_interface` arg will make the factory
        #     # wrap the chosen default model with our new model API class
        #     # (DuelingQModel). This way, both `forward` and `get_q_values`
        #     # are available in the returned class.
        #     model_interface=ContActionQModel
        #     if args.framework != "torch" else TorchContActionQModel,
        #     name="cont_action_q_model",
        # )

    def get_local_dir(self) -> str:
        return f"output_ray/independent_learning/{self.trainer_type.__name__}/"

    def _create_tune_config(self) -> dict:
        return {
            **super()._create_tune_config(),
            "num_samples": 1,
        }

    def _create_model_config(self) -> dict:
        return {
            # Todo: I should probably add the models into the individual policies here!
            **super()._create_model_config(),
            "multiagent": {
                "policies": {
                    pu.Agent.Cont.value: PolicySpec(None, self.env.observation_spaces[pu.Agent.Cont.value], self.env.action_spaces[pu.Agent.Cont.value], None),
                    pu.Agent.Host.value: PolicySpec(None, self.env.observation_spaces[pu.Agent.Host.value], self.env.action_spaces[pu.Agent.Host.value], None),
                },
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
            },
            "callbacks": custom.CustomMetricCallbacks,
            # "num_workers": 6,
        }

    @classmethod
    def _create_env(cls, game: pu.Game) -> MultiAgentEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.agent_indicator_v0(env)

        return custom.RayProbabilityUpdatingEnv(env)

    def predict(self, checkpoint: str):
        trainer = self.trainer_type(config=self._create_model_config())
        trainer.restore(checkpoint)

        # dist_class, dist_dim = ModelCatalog.get_action_dist(self.env.action_spaces[pu.Agent.Cont.value], {})
        # print("dist_class", dist_class)
        # print("dist_dim", dist_dim)
        #
        # # model = ModelCatalog.get_model_v2(self.env.observation_spaces[pu.Agent.Cont.value], self.env.action_spaces[pu.Agent.Cont.value], self.env.action_spaces[pu.Agent.Cont.value], {})
        # # print("model", model)
        #
        # policy_cont = trainer.get_policy(pu.Agent.Cont.value)
        # print("policy_cont dist class", policy_cont.dist_class)
        # # print("policy_cont state", policy_cont.get_state())
        # print(policy_cont.model)
        #
        # # dist = dist_class(model.outputs, model)
        # # print("dist", dist)
        #
        # # action = dist.sample()
        # # print("action", action)

        obs = self.env.reset()
        # unsquash makes no difference now I guess
        actions = {agent.value: trainer.compute_single_action(obs[agent.value], explore=False, policy_id=agent.value) for agent in pu.Agent}
        obs, rewards, dones, infos = self.env.step(actions)
        print(self.game)
