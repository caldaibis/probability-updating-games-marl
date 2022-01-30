from __future__ import annotations

import os
import shutil
from pathlib import Path

import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_marl as marl


def fix_underscores():
    root = Path("../lib_vis/data/gaussian_softmax")
    parts = {
        'algos': [marl.PPO, marl.A2C, marl.TD3, marl.SAC],
        'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
        'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
        'interactions': ['cooperative', 'zero-sum'],
        'order': ['games', 'losses', 'interactions', 'algos'],
    }
    
    for first in parts[parts['order'][0]]:
        for second in parts[parts['order'][1]]:
            for third in parts[parts['order'][2]]:
                for fourth in parts[parts['order'][3]]:
                    original_path = root / f'{first}_{second}_{third}_{fourth}.csv'
                    destination_path = root / f'{first}_{second}_{third.replace("-", "_")}_{fourth}.csv'
                    os.rename(original_path, destination_path)


def flatten_paths():
    dirs = {
        'dirichlet': {
            'algos': [marl.PPO, marl.A2C],
            'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
            'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
            'interactions': ['cooperative', 'zero-sum'],
            'order': ['games', 'losses', 'interactions', 'algos'],
        },
        'gaussian_box_rcar_dist': {
            'algos': [marl.PPO, marl.A2C],
            'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
            'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
            'interactions': ['zero-sum'],
            'order': ['games', 'losses', 'interactions', 'algos'],
        },
        'gaussian_box': {
            'algos': [marl.PPO, marl.A2C, marl.DDPG, marl.TD3, marl.SAC],
            'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
            'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
            'interactions': ['cooperative', 'zero-sum'],
            'order': ['losses', 'games', 'interactions', 'algos'],
        },
    }
    correct_order = ['games', 'losses', 'interactions', 'algos']
    
    data_dir = Path("../lib_vis/data/")
    
    for d in dirs:
        root = data_dir / d
        for first in dirs[d][dirs[d]['order'][0]]:
            for second in dirs[d][dirs[d]['order'][1]]:
                for third in dirs[d][dirs[d]['order'][2]]:
                    for fourth in dirs[d][dirs[d]['order'][3]]:
                        original_path = root / first / second / third / f'{fourth}.csv'
                        if dirs[d]['order'][0] == correct_order[0]:
                            destination_path = root / f'{first}_{second}_{third.replace("-", "_")}_{fourth}.csv'
                        else:
                            destination_path = root / f'{second}_{first}_{third.replace("-", "_")}_{fourth}.csv'
                        shutil.copy(original_path, destination_path)


if __name__ == '__main__':
    # flatten_paths()
    fix_underscores()