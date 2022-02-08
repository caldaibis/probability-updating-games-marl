from __future__ import annotations

import json
from typing import Dict

import numpy as np
import pickle

import src.lib_pu as pu
import src.lib_pu.games as pu_games
from pettingzoo.test import parallel_api_test


def generate_random_matrices(outcome_count: int):
    pos_matrices = [pu.matrix_random_pos(outcome_count) for _ in range(10)]
    neg_matrices = [pu.matrix_random_neg(outcome_count) for _ in range(10)]
    mix_matrices = [pu.matrix_random_mix(outcome_count) for _ in range(10)]
    matrices = {'pos': pos_matrices, 'neg': neg_matrices, 'mix': mix_matrices}
    
    file = open(f'saved_matrices/x={outcome_count}.txt', 'wb')
    pickle.dump(matrices, file)
    file.close()
    

def select_matrix(game: str, loss: str):
    if loss in pu.MATRIX_PREDEFINED:
        return read_predefined_matrix(game, loss)
    if loss == pu.MATRIX_CUSTOM_3:
        return pu.matrix_custom_3()
    elif loss == pu.MATRIX_CUSTOM_3_NEG:
        return pu.matrix_custom_3_neg()
    elif loss == pu.MATRIX_CUSTOM_6:
        return pu.matrix_custom_6()
    elif loss == pu.MATRIX_CUSTOM_6_NEG:
        return pu.matrix_custom_6_neg()
    elif loss == pu.MATRIX_ONES_POS:
        return pu.matrix_ones_pos(pu_games.GAMES[game].get_outcome_count())
    elif loss == pu.MATRIX_ONES_NEG:
        return pu.matrix_ones_neg(pu_games.GAMES[game].get_outcome_count())
    elif loss in pu.MATRIX_RAND:
        return read_random_matrix(game, loss)


def read_random_matrix(game: str, loss: str):
    loss = loss.split('_')
    with open(f'saved_matrices/x={pu_games.GAMES[game].get_outcome_count()}.txt', 'rb') as f:
        matrices = pickle.load(f)
        return matrices[loss[-2]][int(loss[-1])]


def read_predefined_matrix(game: str, loss: str):
    loss = loss.split('_')
    with open(f'saved_matrices/{game}.json', 'r') as f:
        matrices = json.load(f)
        if loss[-2] == 'neg':
            return -1 * np.array(matrices[int(loss[-1])]['matrix'])
        else:
            return np.array(matrices[int(loss[-1])]['matrix'])


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
