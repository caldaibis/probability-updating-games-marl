from __future__ import annotations

from typing import Dict

import numpy as np

import probability_updating as pu
from pettingzoo.test import parallel_api_test


def manual_step(game: pu.Game, actions: Dict[pu.Agent, np.ndarray]):
    print("MANUAL STEP")
    print()
    import supersuit as ss
    env = pu.ProbabilityUpdatingEnv(game)
    env = ss.pad_action_space_v0(env)
    env.reset()

    env.step({a.value: actions[a] for a in actions.keys()})


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
    print(f"Mean loss (cont): {mean_loss[pu.Agent.Cont]}")
    print(f"Mean loss (host): {mean_loss[pu.Agent.Host]}")

    print()
    print(f"Mean entropy (cont): {mean_entropy[pu.Agent.Cont]}")
    print(f"Mean entropy (host): {mean_entropy[pu.Agent.Host]}")

    print()
    print("SIMULATION END")


def test(game: pu.Game, actions: Dict[pu.Agent, np.ndarray]):
    print("MANUAL TEST RUN BEGIN")

    manual_step(game, actions)
    print(game)

    simulate(game, actions)

    print("MANUAL TEST RUN END")
    print()


def environment_api_test(game: pu.Game):
    env = pu.ProbabilityUpdatingEnv(game)

    parallel_api_test(env, num_cycles=100)
