from __future__ import annotations

import logging
import sys
from typing import Dict, Optional

import src.probability_updating as pu
import src.probability_updating.games as games
import src.learning as learning
import src.util as util

import ray

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.ddpg import DDPGTrainer, TD3Trainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.marwil import MARWILTrainer
from ray.rllib.agents.pg import PGTrainer


trainers = [PPOTrainer, A2CTrainer, DDPGTrainer, TD3Trainer, SACTrainer]
hyper_param = {
    PPOTrainer: {
        "batch_mode": "complete_episodes",
        "train_batch_size": 64,
        "sgd_minibatch_size": 8,
        "num_sgd_iter": 1,
        "lr": 8e-5,
        # PPO clip parameter.
        "clip_param": 0.1,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
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

algo_list = {
    # 'ppo': PPOTrainer,
    # 'a2c': A2CTrainer,
    'impala': ImpalaTrainer,
    'marwil': MARWILTrainer,
}

loss_list = [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC]

game_list = {
    games.MontyHall.name(): games.MontyHall,
    games.FairDie.name()  : games.FairDie,
    # games.ExampleC.name() : games.ExampleC,
    # games.ExampleD.name() : games.ExampleD,
    # games.ExampleE.name() : games.ExampleE,
    # games.ExampleF.name() : games.ExampleF,
    # games.ExampleH.name() : games.ExampleH,
}

interaction_list = ['zero-sum', 'cooperative']


def run(args: Optional[Dict[str, str]]):
    if not args:
        args = {
            'algorithm': 'marwil',
            'game_type': 'monty_hall',
            'loss_type': 'randomised_0_1',
            'interaction_type': 'zero-sum',
        }
        
    # Essential configuration
    losses = {
        pu.Agent.Cont: args['loss_type'],
        pu.Agent.Host: args['loss_type'] if args['interaction_type'] == 'cooperative' else args['loss_type'] + '_neg'
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
        ray.init(local_mode=False, logging_level=logging.INFO, log_to_driver=False)  # Running
        # ray.init(local_mode=True, logging_level=logging.INFO, log_to_driver=True)  # Debugging
        
        # Configuration
        t = algo_list[args['algorithm']]
        min_total_time_s = 60
        max_total_time_s = 60

        # Run
        ray_model = learning.ModelWrapper(game, losses, t, hyper_param[t], min_total_time_s, max_total_time_s)
        analysis = None
        # analysis = ray_model.load()
        if not analysis:
            analysis = ray_model.learn(show_figure=False, save_progress=True)
        # ray_model.predict(analysis.best_checkpoint)
            
        ray.shutdown()


if __name__ == '__main__':
    args_keys = ['algorithm', 'game_type', 'loss_type', 'interaction_type']
    
    if len(sys.argv) == 1:
        run(None)
    else:
        run(dict(zip(args_keys, sys.argv[1:])))

