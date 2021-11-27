from __future__ import annotations

from typing import Type, Dict

import numpy as np

import probability_updating as pu
import probability_updating.games as games
import models as baseline_models

import logging

import ray
import run_baselines as baselines
import run_ray
from run_ray import RayModel
from ray.tune import register_env


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
    env = pu.ProbabilityUpdatingEnv(game)
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
        pu.cont(): pu.Loss.zero_one(),
        pu.quiz(): pu.Loss.zero_one()
    }
    game = game_type(losses)

    # ---------------------------------------------------------------

    if False:
        # Manual run configuration
        actions = {
            pu.cont(): games.MontyHall.cont_always_switch(),
            pu.quiz(): games.MontyHall.quiz_uniform()
        }

        # Run
        test(game_type, losses, actions)

    # ---------------------------------------------------------------

    if False:
        # Baseline configuration
        model_type = baseline_models.PPO
        total_timesteps = 10000
        ext_name = ""

        # Run
        baselines.learn(game, losses, model_type, total_timesteps, ext_name)
        game = baselines.predict(game, losses, model_type, total_timesteps, ext_name)
        write_results(game)

    # ---------------------------------------------------------------

    if True:
        # Ray configuration
        env = run_ray.shared_parameter_env(game)
        register_env("pug", lambda _: env)
        ray_model_type = RayModel.PPO
        iterations = 10

        # Run
        # Zet local_mode=True om te debuggen
        ray.init(local_mode=True, logging_level=logging.DEBUG)
        checkpoint = run_ray.learn(game, losses, ray_model_type, iterations)
        run_ray.predict(game, env, ray_model_type, checkpoint)
        write_results(game)
        ray.shutdown()

    # ---------------------------------------------------------------


if __name__ == '__main__':
    run()
