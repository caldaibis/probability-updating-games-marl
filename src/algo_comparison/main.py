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

import visualisation

trainers = [PPOTrainer, A2CTrainer, DDPGTrainer, TD3Trainer, SACTrainer]
hyper_param = {
    PPOTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "sgd_minibatch_size": 8,
        "num_sgd_iter": 1,
        "lr": 8e-5,
    },
    A2CTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "min_iter_time_s": 1,
    },
    DDPGTrainer: {
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
        "train_batch_size": 64,
        "rollout_fragment_length": 64,
        "timesteps_per_iteration": 16,
        "prioritized_replay_beta_annealing_timesteps": 16,
        "learning_starts": 1,
        "min_iter_time_s": 0,
    },
}


def run():
    # Essential configuration
    losses = {
        pu.Agent.Cont: pu.Loss.zero_one(),
        pu.Agent.Host: pu.Loss.zero_one_negative()
    }
    game = pu.games.MontyHall(losses)

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
        ray.init(local_mode=False, logging_level=logging.INFO, log_to_driver=False)  # Running
        # ray.init(local_mode=True, logging_level=logging.INFO, log_to_driver=True)  # Debugging

        t = PPOTrainer

        min_total_time_s = 60
        max_total_time_s = 60

        ray_model = scripts_ray.IndependentLearning(game, losses, t, hyper_param[t], min_total_time_s, max_total_time_s)

        # Run
        analysis = None
        # analysis = ray_model.load()
        if not analysis:
            analysis = ray_model.learn()
            
        ray_model.predict(analysis.best_checkpoint)
        ray_model.save_to_results(analysis)
        visualisation.direct(analysis.trials)

    ray.shutdown()


if __name__ == '__main__':
    run()
