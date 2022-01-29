import subprocess
from typing import Dict, Any, List

import src.pu_lib as pu
import src.pu_lib.games as pu_games
import src.marl_lib as marl
import main


def output_args(_args: Dict[str, Any]) -> List[str]:
    return [str(_args[key]) for key in main.arg_keys]


if __name__ == '__main__':
    args = {
        'debug_mode': False,
        'show_example': True,
        'learn': False,
        'predict': False,
        'show_figure': False,
        'save_progress': False,
    }
    
    for algo in [marl.PPO]:  # marl.ALGOS:
        args['algorithm'] = algo
        for game in pu_games.GAMES:
            args['game'] = game
            for (cont, host) in pu.LOSS_PAIRS_ALL:
                args[pu.CONT] = cont
                args[pu.HOST] = host
                
                output = output_args(args)
                print(output)
                subprocess.call(["../../venv/Scripts/python", 'main.py', *output])
