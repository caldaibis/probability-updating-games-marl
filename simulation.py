from __future__ import annotations

from typing import Dict

import probability_updating as pu
import probability_updating.games as games


def run(game: games.Game, actions: Dict[pu.Agent, pu.StrategyWrapper]):
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
    print(f"Mean loss (cont): {mean_loss[pu.cont()]}")
    print(f"Mean loss (quiz): {mean_loss[pu.quiz()]}")

    print()
    print(f"Mean entropy (cont): {mean_entropy[pu.cont()]}")
    print(f"Mean entropy (quiz): {mean_entropy[pu.quiz()]}")

    print()
    print("SIMULATION END")
