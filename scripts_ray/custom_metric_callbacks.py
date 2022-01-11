from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
from ray.rllib import RolloutWorker, BaseEnv, Policy, SampleBatch
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import Episode

# Custom metrics om toe te voegen:
# - Min rewards over all agents
# - Max rewards over all agents
# - Sum of absolute rewards over all agents
# - Mean of absolute rewards over all agents
from ray.rllib.utils.typing import PolicyID

import probability_updating


class CustomMetricCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy], episode: Episode,
                         **kwargs) -> None:
        print("on_episode_start")
        print("obs_cont", episode.last_observation_for(probability_updating.Agent.Cont.value))
        print("obs_cont", episode.last_observation_for(probability_updating.Agent.Host.value))
        print("raw_obs_cont", episode.last_raw_obs_for(probability_updating.Agent.Cont.value))
        print("raw_obs_host", episode.last_raw_obs_for(probability_updating.Agent.Host.value))

        # print(f"episode {episode.episode_id} started")
        # episode.user_data["policy_rewards"] = []
        # episode.hist_data["policy_rewards"] = []

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None,
                        episode: Episode,
                        **kwargs) -> None:
        pass
        # print("on_episode_step")
        # episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy], episode: Episode,
                       **kwargs) -> None:
        pass
        # print("on_episode_end")
        # pole_angle = np.mean(episode.user_data["pole_angles"])
        #
        # episode.custom_metrics["pole_angle"] = pole_angle
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass
        # print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(
            trainer, result["episodes_this_iter"]))
        pass
        # you can mutate the result dict to add new fields to return
        # result["callback_ok"] = True

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        pass
        # result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
        # print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
        #     policy, result["sum_actions_in_train_batch"]))

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: Episode, agent_id: str,
            policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        # if "num_batches" not in episode.custom_metrics:
        #     episode.custom_metrics["num_batches"] = 0
        # episode.custom_metrics["num_batches"] += 1
