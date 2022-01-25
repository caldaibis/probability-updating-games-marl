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
            for (cont, host) in [(pu.RANDOMISED_ZERO_ONE, pu.RANDOMISED_ZERO_ONE_NEG)]:  # main.loss_zero_sum_list:
                args[pu.CONT] = cont
                args[pu.HOST] = host
                
                output = output_args(args)
                print(output)
                subprocess.call(["../../venv/Scripts/python", 'main.py', *output])
