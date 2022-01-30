from __future__ import annotations

import distutils.util
import logging
import sys
from typing import Dict, Optional, List, Any

import numpy as np
import ray

import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_marl as marl
import src.util as util

from src.main import algo_config


string_arg_keys: List[str] = [
    'algorithm',
    'game',
    pu.CONT,
    pu.HOST,
]

bool_arg_keys: List[str] = [
    'show_example',
    'debug_mode',
    'learn',
    'predict',
    'show_figure',
    'save_progress',
]

arg_keys = [*string_arg_keys, *bool_arg_keys]

# def test():
#     outcomes = {
#         1: 1 / 3,
#         2: 1 / 3,
#         3: 1 / 3,
#     }
#     messages = {
#         1: 1 / 2,
#         2: 1 / 2,
#     }
#     cont = {
#         1: {
#             1: 1,
#             2: 0,
#             3: 0,
#         },
#         2: {
#             1: 0,
#             2: 0,
#             3: 1,
#         },
#     }
#     host = {
#         1: {
#             1: 1,
#             2: 0,
#         },
#         2: {
#             1: 1 / 2,
#             2: 1 / 2,
#         },
#         3: {
#             1: 0,
#             2: 1,
#         },
#     }
#     reverse_host = {
#         1: {
#             1: 2 / 3,
#             2: 1 / 3,
#             3: 0,
#         },
#         2: {
#             1: 0,
#             2: 1 / 3,
#             3: 2 / 3,
#         },
#     }
#
#     loss_fn = pu.LOSS_FNS[pu.RANDOMISED_ZERO_ONE]
#
#     # calc expected loss
#     loss = 0
#     for x in outcomes:
#         _l = 0
#         for y in messages:
#             _l = host[y][x] * loss_fn(cont, outcomes, x, y)
#         _l *= p[x]
#
#
#     # calc expected entropy
#     entropy = 0
#     for y in [1, 2]:
#         e = self.message_dist[y] * self.get_entropy(agent, y)
#         if not math.isnan(e):
#             ent += e
#
#     return ent


def run(args: Optional[Dict[str, Any]]):
    if not args:
        args = {
            'algorithm': marl.PPO,
            'game': pu_games.MONTY_HALL,
            pu.CONT: pu.MATRIX,
            pu.HOST: pu.MATRIX,
            'debug_mode': False,
            'show_example': True,
            'learn': True,
            'predict': True,
            'show_figure': True,
            'save_progress': False,
        }
    else:
        for k in bool_arg_keys:
            args[k] = distutils.util.strtobool(args[k])
        
    # Essential configuration
    losses = {
        pu.CONT: args[pu.CONT],
        pu.HOST: args[pu.HOST],
    }
    
    # def custom_matrix(outcome_count):
    #     return np.array(
    #         [    # x prime
    #             [0, 1, 1],
    #             [1, 0, 1],  # x
    #             [1, 1, 0]
    #         ]
    #     )
    
    if args[pu.CONT] == pu.MATRIX:
        matrix_gen = {
            pu.CONT: pu.matrix_ones_pos,
            pu.HOST: pu.matrix_ones_neg,
        }
        game = pu_games.GAMES[args['game']](losses, matrix_gen)
    else:
        game = pu_games.GAMES[args['game']](losses)

    if args['show_example']:
        util.example_step(
            game,
            {
                pu.CONT: game.cont_optimal_zero_one(),
                pu.HOST: game.host_default(),
            }
        )

    # Configuration
    t = marl.ALGOS[args['algorithm']]
    min_total_time_s = 80
    max_total_time_s = 80
    model_config = algo_config.hyper_parameters[t]
    
    if args['learn']:
        if args['debug_mode']:
            ray.init(local_mode=True, logging_level=logging.DEBUG, log_to_driver=True)
            model_config['num_workers'] = 0
        else:
            ray.init(local_mode=False, logging_level=logging.INFO, log_to_driver=False)
            model_config['num_workers'] = 10
        
        # Run
        model = marl.ModelWrapper('experimental_dirichlet', game, losses, t, model_config, min_total_time_s, max_total_time_s)
        # analysis = model.load()
        model.learn(predict=args['predict'], show_figure=args['show_figure'], save_progress=args['save_progress'])
        
        ray.shutdown()


if __name__ == '__main__':
    # test()
    
    if len(sys.argv) == 1:
        run(None)
    else:
        run(dict(zip(arg_keys, sys.argv[1:])))

