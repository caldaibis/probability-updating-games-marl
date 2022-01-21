from __future__ import annotations

import logging

from ray.rllib.models import ModelCatalog
from ray.rllib.utils.spaces.simplex import Simplex
from torch.distributions import Dirichlet
import torch
from torch.nn.functional import softmax

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
        "exploration_config": {
            # "type": "NormalisedGaussianNoise",
            # "random_timesteps": 10000,
            # "stddev": 0.1,
            # "initial_scale": 1.0,
            # "final_scale": 1.0,
            # "scale_timesteps": 1
        },
        "model": {
            # "custom_action_dist": "multi_categorical_probs",
        }
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
    # concentration = torch.tensor([1.0, 1.0])
    # dirichlet = Dirichlet(concentration)
    # print(dirichlet.sample())
    # print(dirichlet.sample())
    # print(dirichlet.sample())
    # print(dirichlet.sample())
    # print(dirichlet.sample())
    # print("mean", dirichlet.mean)
    # print("softmax", softmax(concentration))
    # print()

    # concentration = torch.tensor([100.0, 100.0])
    # dirichlet2 = Dirichlet(concentration)
    # print(dirichlet2.sample())
    # print(dirichlet2.sample())
    # print(dirichlet2.sample())
    # print(dirichlet2.sample())
    # print(dirichlet2.sample())
    # print("mean", dirichlet2.mean)
    # print("softmax", softmax(concentration))
    # print()
    #
    # space = Simplex((2, 2), concentration=torch.tensor([100.0, 100.0]))
    # print(torch.tensor(space.sample()))
    # print(torch.tensor(space.sample()))
    # print(torch.tensor(space.sample()))
    # print(torch.tensor(space.sample()))
    # print(torch.tensor(space.sample()))
    #
    # space_2 = scripts_ray.Simplex(2, concentration=[100.0, 100.0])
    # print(torch.tensor(space_2.sample()))
    # print(torch.tensor(space_2.sample()))
    # print(torch.tensor(space_2.sample()))
    # print(torch.tensor(space_2.sample()))
    # print(torch.tensor(space_2.sample()))

    # Essential configuration
    losses = {
        pu.Agent.Cont: pu.Loss.logarithmic(),
        pu.Agent.Host: pu.Loss.logarithmic_negative()
    }
    game = pu.games.FairDie(losses)

    if True:
        # Manual configuration
        actions = {
            pu.Agent.Cont: game.cont_optimal_zero_one(),
            pu.Agent.Host: game.host_default()
        }

        # Run
        # util.manual_step(game, actions)
        print(game)

    if True:
        # Configuration
        # ray.init(local_mode=False, logging_level=logging.INFO, log_to_driver=False)  # Running
        ray.init(local_mode=True, logging_level=logging.INFO, log_to_driver=True)  # Debugging

        t = PPOTrainer

        min_total_time_s = 30
        max_total_time_s = 60

        ray_model = scripts_ray.IndependentLearning(game, losses, t, hyper_param[t], min_total_time_s, max_total_time_s)

        # Run
        best = None
        # best = ray_model.load()
        if not best:
            analysis = ray_model.learn()
            best = analysis.best_checkpoint
            visualisation.direct(analysis.trials)
            # ray_model.save_to_results(analysis)

        ray_model.predict(best)
        print(game)

    ray.shutdown()


if __name__ == '__main__':
    run()
