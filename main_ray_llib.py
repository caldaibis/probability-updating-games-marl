from __future__ import annotations

from typing import Type, Dict

from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import probability_updating as pu
import probability_updating.games as games

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl


def ray(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.LossFunc | pu.Loss]):
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN

    register_env("probability_updating", lambda _: ParallelPettingZooEnv(pu.probability_updating_env.env(game_type(losses))))

    tune.run(
        "APEX_DDPG",
        stop={"episodes_total": 60000},
        checkpoint_freq=10,
        config={
            # Enviroment specific.
            "env": "probability_updating",

            # General
            "num_gpus": 1,
            "num_workers": 2,
            "num_envs_per_worker": 8,
            "learning_starts": 1000,
            "buffer_size": int(1e5),
            "compress_observations": True,
            "rollout_fragment_length": 20,
            "train_batch_size": 512,
            "gamma": .99,
            "n_step": 3,
            "lr": .0001,
            "prioritized_replay_alpha": 0.5,
            "final_prioritized_replay_beta": 1.0,
            "target_network_update_freq": 50000,
            "timesteps_per_iteration": 25000,

            # Method specific.
            "multiagent": {
                # We only have one policy (calling it "shared").
                # Class, obs/act-spaces, and config will be derived
                # automatically.
                "policies": {"shared_policy"},
                # Always use "shared" policy.
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: "shared_policy"),
            },
        },
    )
