from __future__ import annotations

from typing import Type, Dict

import numpy as np

import probability_updating as pu
import games
import models as baseline_models

import run_baselines as baselines
import run_ray as ray
from run_ray import RayModel


def write_results(game: games.Game):
    import json

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


def manual(game, actions: Dict[pu.Agent, np.ndarray]):
    import supersuit as ss
    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env.reset()

    env.step(actions)

    return game


def test(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], actions: Dict[pu.Agent, np.ndarray]):
    import simulation

    print("MANUAL TEST RUN BEGIN")
    print()
    print(f"Game: {game_type.name()}")

    game = game_type(losses)
    print(f"Action space (cont): {game.get_cont_action_space()}")
    print(f"Action space (quiz): {game.get_quiz_action_space()}")

    print(f"Loss function (cont): {losses[pu.cont()].name}")
    print(f"Loss function (quiz): {losses[pu.quiz()].name}")

    game = manual(game, actions)
    write_results(game)

    simulation.run(game, actions)

    print("MANUAL TEST RUN END")
    print()


def run():
    # ---------------------------------------------------------------

    # Essential configuration
    game_type = games.MontyHall
    losses = {
        pu.cont(): pu.Loss.logarithmic(),
        pu.quiz(): pu.Loss.logarithmic_negative()
    }

    # ---------------------------------------------------------------

    # Manual run configuration
    actions = {
        pu.cont(): games.MontyHall.cont_always_switch(),
        pu.quiz(): games.MontyHall.quiz_uniform()
    }

    # Run
    test(game_type, losses, actions)

    # ---------------------------------------------------------------

    # Baseline configuration
    model_type = baseline_models.PPO
    total_timesteps = 100000
    ext_name = ""

    # Run
    if False:
        baselines.learn(game_type, losses, model_type, total_timesteps, ext_name)
        game = baselines.predict(game_type, losses, model_type, total_timesteps, ext_name)
        write_results(game)

    # ---------------------------------------------------------------

    # Ray configuration
    ray_model_type = RayModel.MADDPG
    iterations = 5

    # Run
    if True:
        checkpoint = ray.learn(game_type, losses, ray_model_type, iterations)
        game = ray.predict(game_type, losses, ray_model_type, checkpoint)
        write_results(game)

    # ---------------------------------------------------------------


if __name__ == '__main__':
    run()
