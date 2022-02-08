import subprocess
from typing import Dict, Any, List

import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_marl as marl
import main


def output_args(_args: Dict[str, Any]) -> List[str]:
    return [str(_args[key]) for key in main.arg_keys]

    # args = {
    #     'debug_mode': False,
    #     'show_example': False,
    #     'ray': True,
    #     'learn': True,
    #     'expectation_run': True,
    #     'predict': False,
    #     'show_figure': False,
    #     'show_eval': True,
    #     'save_progress': False,
    #     'min_total_time_s': 5,
    #     'max_total_time_s': 5,
    # }
    #
    # for algo in [marl.PPO]:
    #     args['algorithm'] = algo
    #     for game in [pu_games.MONTY_HALL]:
    #         args['game'] = game
    #         for (cont, host) in [(pu.MATRIX_RAND_POS[0], pu.MATRIX_RAND_NEG[0])]:
    #             args[pu.CONT] = cont
    #             args[pu.HOST] = host
    #
    #             output = output_args(args)
    #             print(output)
    #             subprocess.call(["../../venv/Scripts/python", 'main.py', *output])
    #             subprocess.call(["../../venv/Scripts/python", 'main.py', *output])
    #             subprocess.call(["../../venv/Scripts/python", 'main.py', *output])
    #             subprocess.call(["../../venv/Scripts/python", 'main.py', *output])


def run_n_times_and_load(n, game, losses):
    run_args = {
        'algorithm': marl.PPO,
        'game': game,
        pu.CONT: losses[pu.CONT],
        pu.HOST: losses[pu.HOST],
        'debug_mode': False,
        'show_example': True,
        'ray': True,
        'learn': True,
        'expectation_run': True,
        'predict': False,
        'show_figure': False,
        'show_eval': False,
        'save_progress': False,
        'min_total_time_s': 30,
        'max_total_time_s': 30,
    }
    
    for _ in range(n):
        subprocess.call(["../../venv/Scripts/python", 'main.py', *output_args(run_args)])
    
    load_args = {
        'algorithm': marl.PPO,
        'game': game,
        pu.CONT: losses[pu.CONT],
        pu.HOST: losses[pu.HOST],
        'debug_mode': False,
        'show_example': False,
        'ray': True,
        'learn': False,
        'expectation_run': True,
        'predict': True,
        'show_figure': True,
        'show_eval': True,
        'save_progress': False,
        'min_total_time_s': 30,
        'max_total_time_s': 30,
    }
    
    subprocess.call(["../../venv/Scripts/python", 'main.py', *output_args(load_args)])


if __name__ == '__main__':
    for m in pu.MATRIX_PREDEFINED:
        losses = {
            pu.CONT: pu.MATRIX_PREDEFINED[0],
            pu.HOST: m,
        }
        run_n_times_and_load(0, pu_games.MONTY_HALL, losses)
