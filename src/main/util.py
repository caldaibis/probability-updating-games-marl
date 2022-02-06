from __future__ import annotations

from typing import Dict

import numpy as np
import pickle

import src.lib_pu as pu
from pettingzoo.test import parallel_api_test


def generate_random_matrices(outcome_count: int):
    pos_matrices = [pu.matrix_random_pos(outcome_count) for _ in range(10)]
    neg_matrices = [pu.matrix_random_neg(outcome_count) for _ in range(10)]
    mix_matrices = [pu.matrix_random_mix(outcome_count) for _ in range(10)]
    matrices = {'pos': pos_matrices, 'neg': neg_matrices, 'mix': mix_matrices}
    
    file = open(f'saved_matrices/x={outcome_count}.txt', 'wb')
    pickle.dump(matrices, file)
    file.close()
    
    
def read_random_matrix(outcome_count: int, sign: str, idx: int):
    with open(f'saved_matrices/x={outcome_count}.txt', 'rb') as f:
        matrices = pickle.load(f)
        return matrices[sign][idx]


def example_step(game: pu.Game, actions: Dict[pu.Agent, np.ndarray]):
    env = pu.ProbabilityUpdatingEnv(game)
    env.reset()

    env.step(actions)
    print(game)


def simulate(game: pu.Game, actions: Dict[pu.Agent, np.ndarray]):
    print("SIMULATION BEGIN")
    print()
    print("Running simulation...")
    sim = pu.SimulationWrapper(game, actions)

    x_count, y_count, mean_loss, mean_entropy = sim.simulate(100000)

    print()
    for x in x_count.keys():
        print(f"x{x.id}: {x_count[x]} times")

    print()
    for y in y_count.keys():
        print(f"y{y.id}: {y_count[y]} times")

    print()
    print(f"Mean loss (cont): {mean_loss[pu.CONT]}")
    print(f"Mean loss (host): {mean_loss[pu.HOST]}")

    print()
    print(f"Mean entropy (cont): {mean_entropy[pu.CONT]}")
    print(f"Mean entropy (host): {mean_entropy[pu.HOST]}")

    print()
    print("SIMULATION END")


def environment_api_test(game: pu.Game):
    env = pu.ProbabilityUpdatingEnv(game)

    parallel_api_test(env, num_cycles=100)
