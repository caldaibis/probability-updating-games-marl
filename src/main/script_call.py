import subprocess
from typing import Dict, Any, List

import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_marl as marl
import main


def output_args(_args: Dict[str, Any]) -> List[str]:
    return [str(_args[key]) for key in main.arg_keys]


def load(game, losses, t):
    run_args = {
        'algorithm': marl.PPO,
        'game': game,
        pu.CONT: losses[pu.CONT],
        pu.HOST: losses[pu.HOST],
        'debug_mode': False,
        'show_example': False,
        'ray': True,
        'learn': False,
        'show_figure': False,
        'show_eval': True,
        'save_figures': True,
        'save_progress': False,
        'min_total_time_s': t,
        'max_total_time_s': t,
    }
    
    subprocess.call(["../../venv/Scripts/python", 'main.py', *output_args(run_args)])


def learn(n, game, losses, t):
    run_args = {
        'algorithm': marl.PPO,
        'game': game,
        pu.CONT: losses[pu.CONT],
        pu.HOST: losses[pu.HOST],
        'debug_mode': False,
        'show_example': False,
        'ray': True,
        'learn': True,
        'show_figure': False,
        'show_eval': True,
        'save_figures': True,
        'save_progress': False,
        'min_total_time_s': t,
        'max_total_time_s': t,
    }
    
    for _ in range(n):
        subprocess.call(["../../venv/Scripts/python", 'main.py', *output_args(run_args)])


def show_matrices(game):
    for m in pu.MATRIX_PREDEFINED_POS:
        run_args = {
            'algorithm': marl.PPO,
            'game': game,
            pu.CONT: m,
            pu.HOST: m,
            'debug_mode': False,
            'show_example': True,
            'ray': False,
            'learn': True,
            'show_figure': False,
            'show_eval': True,
            'save_figures': True,
            'save_progress': False,
            'min_total_time_s': 10,
            'max_total_time_s': 10,
        }
        subprocess.call(["../../venv/Scripts/python", 'main.py', *output_args(run_args)])


def learn_predefined_matrices(game, t):
    # Cont: Pos 0, Host: Pos n
    for m in pu.MATRIX_PREDEFINED_POS:
        losses = {
            pu.CONT: pu.MATRIX_PREDEFINED_POS[0],
            pu.HOST: m,
        }
        learn(1, game, losses, t)

    # Cont: Pos 0, Host: Neg n
    for m in pu.MATRIX_PREDEFINED_NEG:
        losses = {
            pu.CONT: pu.MATRIX_PREDEFINED_POS[0],
            pu.HOST: m,
        }
        learn(1, game, losses, t)

    # Cont: Pos n, Host: Pos 0
    for m in pu.MATRIX_PREDEFINED_POS:
        losses = {
            pu.CONT: m,
            pu.HOST: pu.MATRIX_PREDEFINED_POS[0],
        }
        learn(1, game, losses, t)

    # Cont: Pos n, Host: Neg 0
    for m in pu.MATRIX_PREDEFINED_POS:
        losses = {
            pu.CONT: m,
            pu.HOST: pu.MATRIX_PREDEFINED_NEG[0],
        }
        learn(1, game, losses, t)

    # Cont: Pos n, Host: Pos n
    for m in pu.MATRIX_PREDEFINED_POS:
        losses = {
            pu.CONT: m,
            pu.HOST: m,
        }
        learn(1, game, losses, t)

    # Cont: Pos n, Host: Neg n
    for (pos_m, neg_m) in zip(pu.MATRIX_PREDEFINED_POS, pu.MATRIX_PREDEFINED_NEG):
        losses = {
            pu.CONT: pos_m,
            pu.HOST: neg_m,
        }
        learn(1, game, losses, t)
        

def learn_predefined_matrix(game, t, i):
    # Cont: Pos 0, Host: Pos n
    losses = {
        pu.CONT: pu.MATRIX_PREDEFINED_POS[0],
        pu.HOST: pu.MATRIX_PREDEFINED_POS[i],
    }
    learn(1, game, losses, t)

    # Cont: Pos 0, Host: Neg n
    losses = {
        pu.CONT: pu.MATRIX_PREDEFINED_POS[0],
        pu.HOST: pu.MATRIX_PREDEFINED_NEG[i],
    }
    learn(1, game, losses, t)

    if i != 0:
        # Cont: Pos n, Host: Pos 0
        losses = {
            pu.CONT: pu.MATRIX_PREDEFINED_POS[i],
            pu.HOST: pu.MATRIX_PREDEFINED_POS[0],
        }
        learn(1, game, losses, t)
    
        # Cont: Pos n, Host: Neg 0
        losses = {
            pu.CONT: pu.MATRIX_PREDEFINED_POS[i],
            pu.HOST: pu.MATRIX_PREDEFINED_NEG[0],
        }
        learn(1, game, losses, t)
    
        # Cont: Pos n, Host: Pos n
        losses = {
            pu.CONT: pu.MATRIX_PREDEFINED_POS[i],
            pu.HOST: pu.MATRIX_PREDEFINED_POS[i],
        }
        learn(1, game, losses, t)
    
        # Cont: Pos n, Host: Neg n
        losses = {
            pu.CONT: pu.MATRIX_PREDEFINED_POS[i],
            pu.HOST: pu.MATRIX_PREDEFINED_NEG[i],
        }
        learn(1, game, losses, t)


if __name__ == '__main__':
    # Example H - 0,4,7,10,13
    # for i in [0, 4, 7, 10, 13]:
    #     learn_predefined_matrix(pu_games.EXAMPLE_H, 80, i)
    
    # Example C & Example G - 0 t/m 7
    for g in [pu_games.EXAMPLE_C, pu_games.EXAMPLE_G]:
        for i in range(8):
            learn_predefined_matrix(g, 5, i)
    
    # Example D & Example E - 0 t/m 7
    for g in [pu_games.EXAMPLE_D, pu_games.EXAMPLE_E]:
        for i in range(8):
            learn_predefined_matrix(g, 80, i)
