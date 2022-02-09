import subprocess
from typing import Dict, Any, List

import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_marl as marl
import main


def output_args(_args: Dict[str, Any]) -> List[str]:
    return [str(_args[key]) for key in main.arg_keys]


def run_n_times(n, game, losses, loading_run: bool):
    run_args = {
        'algorithm': marl.PPO,
        'game': game,
        pu.CONT: losses[pu.CONT],
        pu.HOST: losses[pu.HOST],
        'debug_mode': False,
        'show_example': False,
        'ray': True,
        'learn': True,
        'expectation_run': True,
        'predict': True,
        'show_figure': False,
        'show_eval': True,
        'save_figures': True,
        'save_progress': False,
        'min_total_time_s': 40,
        'max_total_time_s': 40,
    }
    
    for _ in range(n):
        subprocess.call(["../../venv/Scripts/python", 'main.py', *output_args(run_args)])
    
    if loading_run:
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
            'show_figure': False,
            'show_eval': True,
            'save_figures': True,
            'save_progress': False,
            'min_total_time_s': 40,
            'max_total_time_s': 40,
        }
        
        subprocess.call(["../../venv/Scripts/python", 'main.py', *output_args(load_args)])


if __name__ == '__main__':
    for m in pu.MATRIX_PREDEFINED_POS:
        losses = {
            pu.CONT: m,
            pu.HOST: pu.MATRIX_PREDEFINED_POS[0],
        }
        run_n_times(1, pu_games.MONTY_HALL, losses, loading_run=False)
        
    for m in pu.MATRIX_PREDEFINED_NEG:
        losses = {
            pu.CONT: m,
            pu.HOST: pu.MATRIX_PREDEFINED_POS[0],
        }
        run_n_times(1, pu_games.MONTY_HALL, losses, loading_run=False)
        
    # for m in pu.MATRIX_PREDEFINED_POS:
    #     losses = {
    #         pu.CONT: m,
    #         pu.HOST: m,
    #     }
    #     run_n_times(0, pu_games.MONTY_HALL, losses, loading_run=True)
    #
    # for m in pu.MATRIX_PREDEFINED_NEG:
    #     losses = {
    #         pu.CONT: m,
    #         pu.HOST: m,
    #     }
    #     run_n_times(0, pu_games.MONTY_HALL, losses, loading_run=True)
