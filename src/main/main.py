from __future__ import annotations

import distutils.util
import logging
import sys
from typing import Dict, Optional, List, Any

import ray

import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_marl as marl
import src.main.util as util

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
    'ray',
    'learn',
    'predict',
    'show_figure',
    'show_eval',
    'save_figures',
    'save_progress',
]

int_arg_keys: List[str] = [
    'min_total_time_s',
    'max_total_time_s',
]

arg_keys = [*string_arg_keys, *bool_arg_keys, *int_arg_keys]


def run(args: Optional[Dict[str, Any]]):
    if not args:
        args = {
            'algorithm': marl.PPO,
            'game': pu_games.MONTY_HALL,
            pu.CONT: pu.MATRIX_PREDEFINED_POS[3],
            pu.HOST: pu.MATRIX_PREDEFINED_NEG[0],
            'debug_mode': False,
            'show_example': True,
            'ray': True,
            'learn': True,
            'predict': True,
            'show_figure': False,
            'show_eval': True,
            'save_figures': True,
            'save_progress': False,
            'min_total_time_s': 100,
            'max_total_time_s': 100,
        }
    else:
        for k in bool_arg_keys:
            args[k] = distutils.util.strtobool(args[k])
        for k in int_arg_keys:
            args[k] = int(args[k])
        
    # Essential configuration
    losses = {
        pu.CONT: args[pu.CONT],
        pu.HOST: args[pu.HOST],
    }
    
    matrix = {}
    for agent in pu.AGENTS:
        if args[agent].startswith(pu.MATRIX):
            matrix[agent] = util.select_matrix(args['game'], args[agent])
    
    game = pu_games.GAMES[args['game']](losses, matrix)

    if args['show_example']:
        util.example_step(
            game,
            {
                pu.CONT: game.cont_x2_x1_x0(),
                pu.HOST: game.host_always_y2(),
            }
        )

    # Configuration
    algo = marl.ALGOS[args['algorithm']]
    model_config = algo_config.hyper_parameters[algo]
    tune_config = {}
    
    if args['ray']:
        if args['debug_mode']:
            ray.init(local_mode=True, logging_level=logging.INFO, log_to_driver=True)
            tune_config["verbose"] = 3
            model_config['num_workers'] = 0
            tune_config['num_samples'] = 1
        else:
            ray.init(local_mode=False, logging_level=logging.INFO, log_to_driver=False)
            tune_config["verbose"] = 1
            model_config['num_workers'] = 1
            tune_config['num_samples'] = 5
        
        # Run
        config = {
            'tune_config': tune_config,
            'model_config': model_config,
            'min_total_time_s': args['min_total_time_s'],
            'max_total_time_s': args['max_total_time_s'],
            'save_figures': args['save_figures'],
        }
        
        model = marl.ModelWrapper('experimental_dirichlet', game, losses, algo, config)
        model(learn=args['learn'], predict=args['predict'], show_figure=args['show_figure'], show_eval=args['show_eval'], save_progress=args['save_progress'])
        
        ray.shutdown()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run(None)
    else:
        run(dict(zip(arg_keys, sys.argv[1:])))
