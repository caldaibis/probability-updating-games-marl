import subprocess

import main

if __name__ == '__main__':
    for algo in main.algo_list:
        for game in main.game_list:
            for loss in main.loss_list:
                for interaction in main.interaction_list:
                    print(f'START: running {algo} on {interaction} {game} with {loss} loss')
                    subprocess.call(["../../venv/Scripts/python", 'main.py', algo, game, loss, interaction])
                    print(f'END: running {algo} on {interaction} {game} with {loss} loss')
