import subprocess
from typing import Dict, Any, List

import src.probability_updating as pu
import main


def output_args(_args: Dict[str, Any]) -> List[str]:
    return [str(_args[key]) for key in main.args_keys]


if __name__ == '__main__':
    args = {
        'show_example': False,
        'debug_mode': False,
    }
    
    for algo in main.algo_list:
        args['algorithm'] = algo
        for game in main.game_list:
            args['game'] = game
            for (cont, host) in main.loss_zero_sum_list:
                args[pu.CONT] = cont
                args[pu.HOST] = host
                print(f'START: running {args["algorithm"]} on {args["game"]} with {args[pu.CONT]} + {args[pu.HOST]} loss')
                subprocess.call(["../../venv/Scripts/python", 'main.py', *output_args(args)])
                print(f'END: ran {args["algorithm"]} on {args["game"]} with {args[pu.CONT]} + {args[pu.HOST]} loss')
