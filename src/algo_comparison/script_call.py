import subprocess

import main

if __name__ == '__main__':
    for algo in main.algo_list:
        for game in main.game_list:
            for (cont, host) in main.loss_pair_list:
                print(f'{algo} + {game} + {cont} + {host}')
                subprocess.call(["../../venv/Scripts/python", "main.py", algo, game, cont, host])
