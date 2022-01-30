from __future__ import annotations

import distutils.util
import logging
import sys
from typing import Dict, Optional, List, Any

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


def run(args: Optional[Dict[str, Any]]):
    if not args:
        args = {
            'algorithm': marl.PPO,
            'game': pu_games.FAIR_DIE,
            pu.CONT: pu.RANDOMISED_ZERO_ONE,
            pu.HOST: pu.RANDOMISED_ZERO_ONE_NEG,
            'debug_mode': False,
            'show_example': True,
            'learn': True,
            'predict': False,
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
    
    if args[pu.CONT] == pu.MATRIX:
        matrix_gen = {
            pu.CONT: pu.matrix_ones,
            pu.HOST: pu.matrix_ones_neg,
        }
        game = pu_games.GAMES[args['game']](losses, matrix_gen)
    else:
        game = pu_games.GAMES[args['game']](losses)

    if args['show_example']:
        util.example_step(game, {pu.CONT: game.cont_optimal_zero_one(), pu.HOST: game.host_default()})

    # Configuration
    t = marl.ALGOS[args['algorithm']]
    min_total_time_s = 100
    max_total_time_s = 100
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
    if len(sys.argv) == 1:
        run(None)
    else:
        run(dict(zip(arg_keys, sys.argv[1:])))

