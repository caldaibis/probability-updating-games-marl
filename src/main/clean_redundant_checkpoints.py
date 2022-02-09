from __future__ import annotations

import os
import shutil
from pathlib import Path


def run():
    root = Path(f'output_ray/PPO/monty_hall/')
    for experiment in [f.path for f in os.scandir(root) if f.is_dir()]:
        print(experiment)
        for trial in [f.path for f in os.scandir(experiment) if f.is_dir()]:
            print(trial)
            for checkpoint in [f.path for f in os.scandir(trial) if f.is_dir()][:-1]:
                print(checkpoint)
                shutil.rmtree(checkpoint)
                

if __name__ == '__main__':
    run()
