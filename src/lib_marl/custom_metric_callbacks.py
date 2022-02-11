from __future__ import annotations

from typing import Dict, Optional

from ray.rllib import RolloutWorker, BaseEnv, Policy, SampleBatch
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import PolicyID

import src.lib_pu as pu
import src.lib_marl as marl


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
        episode.custom_metrics["reward_cont"] = episode.last_reward_for(pu.CONT)
        episode.custom_metrics["reward_host"] = episode.last_reward_for(pu.HOST)
        
        episode.custom_metrics["reward_min"] = min(episode.custom_metrics["reward_cont"], episode.custom_metrics["reward_host"])
        episode.custom_metrics["reward_max"] = max(episode.custom_metrics["reward_cont"], episode.custom_metrics["reward_host"])
        
        episode.custom_metrics["universal_reward"] = episode.custom_metrics["reward_cont"] + episode.custom_metrics["reward_host"] - (episode.custom_metrics["reward_max"] - episode.custom_metrics["reward_min"])
        
        for y in episode.last_info_for(pu.CONT)["action"]:
            for x in y.outcomes:
                episode.custom_metrics[f'{pu.CONT}_{y}_{x}'] = episode.last_info_for(pu.CONT)["action"][y][x]
        
        for x in episode.last_info_for(pu.HOST)["action"]:
            for y in x.messages:
                episode.custom_metrics[f'{pu.HOST}_{x}_{y}'] = episode.last_info_for(pu.HOST)["action"][x][y]
        
        episode.custom_metrics["expected_entropy"] = episode.last_info_for(pu.CONT)["expected_entropy"]
        episode.custom_metrics["rcar_dist"] = episode.last_info_for(pu.HOST)["rcar_dist"]
        
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result[marl.REWARD_CONT] = result["custom_metrics"][marl.REWARD_CONT]
        result[marl.REWARD_CONT_EVAL] = result["evaluation"]["custom_metrics"][marl.REWARD_CONT]
        result[marl.REWARD_HOST] = result["custom_metrics"][marl.REWARD_HOST]
        result[marl.REWARD_HOST_EVAL] = result["evaluation"]["custom_metrics"][marl.REWARD_HOST]
        
        result["universal_reward_mean"] = result["custom_metrics"]["universal_reward_mean"]
        
        result[marl.RCAR_DIST] = result["custom_metrics"][marl.RCAR_DIST]
        result[marl.RCAR_DIST_EVAL] = result["evaluation"]["custom_metrics"][marl.RCAR_DIST]
        
        result[marl.EXP_ENTROPY] = result["custom_metrics"][marl.EXP_ENTROPY]
        result[marl.EXP_ENTROPY_EVAL] = result["evaluation"]["custom_metrics"][marl.EXP_ENTROPY]

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        pass

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: Episode, agent_id: str,
            policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
