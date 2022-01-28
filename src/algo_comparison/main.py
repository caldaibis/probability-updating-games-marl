from __future__ import annotations

import logging
import sys
from typing import Optional, Dict

from torch.distributions import Categorical, Dirichlet

import util

import probability_updating as pu
import probability_updating.games as games

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
        "clip_param": 0.1,
        "vf_clip_param": 0.1,
    },
    A2CTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "microbatch_size": 8,
        "min_iter_time_s": 0,
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

algo_list = {
    'ppo': PPOTrainer,
    'a2c': A2CTrainer,
    'td3': TD3Trainer,
    'sac': SACTrainer,
}

loss_list = {
    pu.Loss.zero_one().name: pu.Loss.zero_one(),
    pu.Loss.zero_one_negative().name: pu.Loss.zero_one_negative(),
    pu.Loss.brier().name: pu.Loss.brier(),
    pu.Loss.brier_negative().name: pu.Loss.brier_negative(),
    pu.Loss.logarithmic().name: pu.Loss.logarithmic(),
    pu.Loss.logarithmic_negative().name: pu.Loss.logarithmic_negative(),
}

loss_pair_list = [
    (pu.Loss.zero_one().name, pu.Loss.zero_one().name),
    # (pu.Loss.zero_one().name, pu.Loss.zero_one_negative().name),
    
    (pu.Loss.brier().name, pu.Loss.brier().name),
    # (pu.Loss.brier().name, pu.Loss.brier_negative().name),
    
    (pu.Loss.logarithmic().name, pu.Loss.logarithmic().name),
    # (pu.Loss.logarithmic().name, pu.Loss.logarithmic_negative().name),
]

game_list = {
    games.MontyHall.name(): games.MontyHall,
    # games.FairDie.name(): games.FairDie,
}

# games.ExampleC
# games.ExampleD
# games.ExampleE
# games.ExampleF
# games.ExampleH


def run(args: Optional[Dict[str, str]]):
    if not args:
        args = {
            'algorithm': 'ppo',
            'game_type': games.FairDie.name(),
            'cont': pu.Loss.brier().name,
            'host': pu.Loss.brier_negative().name,
        }
    
    # Essential configuration
    losses = {
        pu.Agent.Cont: loss_list[args['cont']],
        pu.Agent.Host: loss_list[args['host']],
    }
    game = game_list[args['game_type']](losses)

    if False:
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

        t = algo_list[args['algorithm']]

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
        # visualisation.direct(analysis.trials)

    ray.shutdown()


if __name__ == '__main__':
    args_keys = ['algorithm', 'game_type', 'cont', 'host']
    
    if len(sys.argv) == 1:
        run(None)
    else:
        run(dict(zip(args_keys, sys.argv[1:])))
