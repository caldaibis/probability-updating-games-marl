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
    'expectation_run',
    'predict',
    'show_figure',
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
            'game': pu_games.FAIR_DIE,
            pu.CONT: pu.MATRIX,
            pu.HOST: pu.MATRIX,
            'debug_mode': False,
            'show_example': True,
            'learn': True,
            'expectation_run': True,
            'predict': True,
            'show_figure': True,
            'save_progress': False,
            'min_total_time_s': 60,
            'max_total_time_s': 5,
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
    
    if args[pu.CONT] == pu.MATRIX:
        outcomes = pu_games.GAMES[args['game']].get_outcome_count()
        m_cont = pu.matrix_ones_pos(outcomes)
        m_host = pu.matrix_ones_neg(outcomes)
        m_host[1][0] *= 2
        matrix = {
            pu.CONT: m_cont,
            pu.HOST: m_host,
        }
        game = pu_games.GAMES[args['game']](losses, matrix)
    else:
        game = pu_games.GAMES[args['game']](losses)

    if args['show_example']:
        util.example_step(
            game,
            {
                pu.CONT: game.cont_optimal_matrix_ones_neg2(),
                pu.HOST: game.host_default(),
            }
        )

    # Configuration
    algo = marl.ALGOS[args['algorithm']]
    model_config = algo_config.hyper_parameters[algo]
    tune_config = {}
    
    if args['learn']:
        if args['debug_mode']:
            ray.init(local_mode=True, logging_level=logging.DEBUG, log_to_driver=True)
            model_config['num_workers'] = 0
            tune_config['num_samples'] = 1
        else:
            ray.init(local_mode=False, logging_level=logging.INFO, log_to_driver=False)
            if args['expectation_run']:
                model_config['num_workers'] = 0
                tune_config['num_samples'] = 12
            else:
                model_config['num_workers'] = 11
                tune_config['num_samples'] = 1
        
        # Run
        config = {
            'tune_config': tune_config,
            'model_config': model_config,
            'min_total_time_s': args['min_total_time_s'],
            'max_total_time_s': args['max_total_time_s'],
        }
        model = marl.ModelWrapper('experimental_dirichlet', game, losses, algo, config)
        # analysis = model.load()
        model.learn(predict=args['predict'], show_figure=args['show_figure'], save_progress=args['save_progress'])
        
        ray.shutdown()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run(None)
    else:
        run(dict(zip(arg_keys, sys.argv[1:])))

