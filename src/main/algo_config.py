from __future__ import annotations

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.ddpg import DDPGTrainer, TD3Trainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.marwil import MARWILTrainer
from ray import tune

hyper_parameters = {
    PPOTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "sgd_minibatch_size": 8,
        "num_sgd_iter": 1,
        "lr": 8e-5,
        "clip_param": 0.1,
        "vf_clip_param": 0.1,
    },
    A2CTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "min_iter_time_s": 1,
    },
    DDPGTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "rollout_fragment_length": 64,
        "evaluation_num_episodes": 1,
        "exploration_config": {
            "random_timesteps": 40,
        },
        "timesteps_per_iteration": 40,
        "replay_buffer_config": {
            "capacity": 400,
        },
        "prioritized_replay_beta_annealing_timesteps": 80,
        "learning_starts": 40,
    },
    TD3Trainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "rollout_fragment_length": 64,
        "evaluation_num_episodes": 1,
        "exploration_config": {
            "random_timesteps": 40,
        },
        "timesteps_per_iteration": 40,
        "replay_buffer_config": {
            "capacity": 400,
        },
        "prioritized_replay_beta_annealing_timesteps": 80,
        "learning_starts": 40,
        "buffer_size": 1000,
    },
    SACTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "rollout_fragment_length": 64,
        "timesteps_per_iteration": 16,
        "prioritized_replay_beta_annealing_timesteps": 16,
        "learning_starts": 1,
        "min_iter_time_s": 0,
    },
    ImpalaTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "min_iter_time_s": 0,
    },
    MARWILTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
    },
}
