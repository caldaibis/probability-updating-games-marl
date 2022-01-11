from __future__ import annotations

import math
from typing import List, Dict, Optional

from ray.rllib import RolloutWorker, BaseEnv, Policy, SampleBatch
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode

# Custom metrics om toe te voegen:
# - Min rewards over all agents
# - Max rewards over all agents
# - Sum of absolute rewards over all agents
# - Mean of absolute rewards over all agents
from ray.rllib.utils.typing import PolicyID

import probability_updating as pu


class CustomMetricCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy], episode: Episode,
                         **kwargs) -> None:
        pass

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None,
                        episode: Episode, **kwargs) -> None:
        pass

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy], episode: Episode,
                       **kwargs) -> None:
        pass

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["policy_reward_mean_cont"] = result["policy_reward_mean"][pu.Agent.Cont.value]
        result["policy_reward_mean_host"] = result["policy_reward_mean"][pu.Agent.Host.value]
        result["policy_reward_mean_min"] = min(result["policy_reward_mean"][pu.Agent.Cont.value], result["policy_reward_mean"][pu.Agent.Host.value])
        result["policy_reward_mean_max"] = max(result["policy_reward_mean"][pu.Agent.Cont.value], result["policy_reward_mean"][pu.Agent.Host.value])
        result["policy_reward_mean_diff"] = -abs(result["policy_reward_mean"][pu.Agent.Cont.value] - result["policy_reward_mean"][pu.Agent.Host.value])
        result["surrogate_reward_mean"] = result["episode_reward_mean"] + result["policy_reward_mean_diff"]

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        pass

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: Episode, agent_id: str,
            policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
