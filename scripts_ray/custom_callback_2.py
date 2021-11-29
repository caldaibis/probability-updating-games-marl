from __future__ import annotations

from typing import Dict, Optional

from ray.rllib import Policy, SampleBatch, BaseEnv, RolloutWorker
from ray.rllib.agents import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import AgentID, PolicyID


class CustomCallback2(DefaultCallbacks):
    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode, env_index: Optional[int] = None, **kwargs) -> None:
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                 env_index=env_index, **kwargs)

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv,
                        policies: Optional[Dict[PolicyID, Policy]] = None, episode: MultiAgentEpisode,
                        env_index: Optional[int] = None, **kwargs) -> None:
        super().on_episode_step(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                env_index=env_index, **kwargs)

    def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, env_index: Optional[int] = None, **kwargs) -> None:
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode,
                               env_index=env_index, **kwargs)

    def on_postprocess_trajectory(self, *, worker: "RolloutWorker", episode: MultiAgentEpisode, agent_id: AgentID,
                                  policy_id: PolicyID, policies: Dict[PolicyID, Policy],
                                  postprocessed_batch: SampleBatch, original_batches: Dict[AgentID, SampleBatch],
                                  **kwargs) -> None:
        super().on_postprocess_trajectory(worker=worker, episode=episode, agent_id=agent_id, policy_id=policy_id,
                                          policies=policies, postprocessed_batch=postprocessed_batch,
                                          original_batches=original_batches, **kwargs)

    def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs) -> None:
        super().on_sample_end(worker=worker, samples=samples, **kwargs)

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
        super().on_learn_on_batch(policy=policy, train_batch=train_batch, result=result, **kwargs)

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        super().on_train_result(trainer=trainer, result=result, **kwargs)
