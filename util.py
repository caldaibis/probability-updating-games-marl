from __future__ import annotations

from typing import Type, Dict

import numpy as np

import probability_updating as pu
import probability_updating.games as games
from pettingzoo.test import parallel_api_test

import json


def write_results(game: games.Game):
    print("Action (cont)", json.dumps(game.get_cont_readable(), indent=2, default=str))
    print("Action (quiz)", json.dumps(game.get_quiz_readable(), indent=2, default=str))
    print()

    print(f"Is CAR?  {game.strategy.is_car()}")
    print(f"Is RCAR? {game.strategy.is_rcar()}")
    print()

    print("Expected loss (cont)", game.get_expected_loss(pu.cont()))
    print("Expected loss (quiz)", game.get_expected_loss(pu.quiz()))
    print()

    print("Expected entropy (cont)", game.get_expected_entropy(pu.cont()))
    print("Expected entropy (quiz)", game.get_expected_entropy(pu.quiz()))
    print()


def manual_step(game, actions: Dict[pu.Agent, np.ndarray]):
    import supersuit as ss
    env = pu.ProbabilityUpdatingEnv(game)
    env = ss.pad_action_space_v0(env)
    env.reset()

    env.step(actions)

    return game


def simulate(game: games.Game, actions: Dict[pu.Agent, np.ndarray]):
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


def test(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], actions: Dict[pu.Agent, np.ndarray]):
    print("MANUAL TEST RUN BEGIN")
    print()
    print(f"Game: {game_type.name()}")

    game = game_type(losses)
    print(f"Action space (cont): {game.get_cont_action_space()}")
    print(f"Action space (quiz): {game.get_quiz_action_space()}")

    print(f"Loss function (cont): {losses[pu.cont()].name}")
    print(f"Loss function (quiz): {losses[pu.quiz()].name}")

    game = manual_step(game, actions)
    write_results(game)

    simulate(game, actions)

    print("MANUAL TEST RUN END")
    print()


def environment_api_test(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss]):
    game = game_type(losses)
    env = pu.ProbabilityUpdatingEnv(game)

    parallel_api_test(env, num_cycles=100)
