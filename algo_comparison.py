from __future__ import annotations

import logging

import util

import probability_updating as pu

import ray
import scripts_ray

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.ddpg import DDPGTrainer, TD3Trainer
from ray.rllib.agents.sac import SACTrainer

trainers = [PPOTrainer, A2CTrainer, DDPGTrainer, TD3Trainer, SACTrainer]
hyper_param = {
    PPOTrainer: {
        "train_batch_size": 64,  # tune.grid_search([64, 96, 128]),
        "sgd_minibatch_size": 8,  # tune.grid_search([4, 8, 16]),
        "num_sgd_iter": 1,
    },
    A2CTrainer: {
        "min_iter_time_s": 0,
    },
    DDPGTrainer: {
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
        "train_batch_size": 64,
        # "actor_hiddens": [400, 300], ??
        # "critic_hiddens": [400, 300], ??
    },
    TD3Trainer: {
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
        "train_batch_size": 64,
        "buffer_size": 1000,
        # "actor_hiddens": [400, 300], ??
        # "critic_hiddens": [400, 300], ??
    },
    SACTrainer: {
        # "Q_model": {
        #     # "fcnet_hiddens": [256, 256], ??
        # }
        # "policy_model": {
        #     "fcnet_hiddens": [256, 256], ??
        # }

        "timesteps_per_iteration": 10,
        "replay_buffer_config": {
            "type": "LocalReplayBuffer",
            "capacity": 1000,
        },
        "prioritized_replay_beta_annealing_timesteps": 200,
        "learning_starts": 40,
        "train_batch_size": 64,
        "min_iter_time_s": 0,
    },
}


def run():
    # Essential configuration
    losses = {
        pu.Agent.Cont: pu.Loss.logarithmic(),
        pu.Agent.Host: pu.Loss.logarithmic()
    }
    game = pu.games.FairDie(losses)

    if True:
        # Manual configuration
        actions = {
            pu.Agent.Cont: game.cont_optimal_zero_one(),
            pu.Agent.Host: game.host_default()
        }

        # Run
        util.manual_step(game, actions)
        print(game)

    if True:
        # Configuration
        ray.init(local_mode=False, logging_level=logging.INFO, log_to_driver=False)  # Zet local_mode=True om te debuggen
        t = PPOTrainer
        timeout_seconds = 300.0

        print()
        print("NOW USING " + t.__name__)
        print()

        ray_model = scripts_ray.ParameterSharingModel(game, losses, t)

        # Run
        best = ray_model.learn(timeout_seconds, hyper_param[t])
        ray_model.predict(best)
        print(game)

    ray.shutdown()


if __name__ == '__main__':
    run()
