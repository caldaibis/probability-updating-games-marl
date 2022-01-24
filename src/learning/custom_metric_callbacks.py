from __future__ import annotations

from typing import List, Dict, Optional

from ray.rllib import RolloutWorker, BaseEnv, Policy, SampleBatch
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID

import src.probability_updating as pu


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
        episode.custom_metrics["reward_cont"] = episode.last_reward_for(pu.Agent.Cont.value)
        episode.custom_metrics["reward_host"] = episode.last_reward_for(pu.Agent.Host.value)
        
        episode.custom_metrics["reward_min"] = min(episode.custom_metrics["reward_cont"], episode.custom_metrics["reward_host"])
        episode.custom_metrics["reward_max"] = max(episode.custom_metrics["reward_cont"], episode.custom_metrics["reward_host"])
        
        episode.custom_metrics["universal_reward"] = episode.custom_metrics["reward_cont"] + episode.custom_metrics["reward_host"] - (episode.custom_metrics["reward_max"] - episode.custom_metrics["reward_min"])
        
        episode.custom_metrics["rcar_rmse"] = episode.last_info_for(pu.Agent.Host.value)["rcar_rmse"]
        
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["reward_cont_mean"] = result["custom_metrics"]["reward_cont_mean"]
        result["reward_cont_eval_mean"] = result["evaluation"]["custom_metrics"]["reward_cont_mean"]
        result["reward_host_mean"] = result["custom_metrics"]["reward_host_mean"]
        result["reward_host_eval_mean"] = result["evaluation"]["custom_metrics"]["reward_host_mean"]
        
        result["universal_reward_mean"] = result["custom_metrics"]["universal_reward_mean"]
        result["universal_reward_eval_mean"] = result["evaluation"]["custom_metrics"]["universal_reward_mean"]
        
        result["rcar_rmse_mean"] = result["custom_metrics"]["rcar_rmse_mean"]
        result["rcar_rmse_eval_mean"] = result["evaluation"]["custom_metrics"]["rcar_rmse_mean"]

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        pass

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: Episode, agent_id: str,
            policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
