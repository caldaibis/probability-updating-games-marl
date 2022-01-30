from __future__ import annotations

from typing import Dict

import numpy as np

import src.lib_pu as pu
from pettingzoo.test import parallel_api_test


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
