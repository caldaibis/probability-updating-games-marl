from __future__ import annotations

import logging

import util

import probability_updating as pu
import probability_updating.games as games

import scripts_baselines

import ray
from ray.tune import register_env
import scripts_ray


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

    if True:
        # Manual run configuration
        actions = {
            pu.cont(): games.MontyHall.cont_always_switch(),
            pu.quiz(): games.MontyHall.quiz_uniform()
        }

        # Run
        util.test(game_type, losses, actions)

    # ---------------------------------------------------------------

    if True:
        # Baseline configuration
        model_type = scripts_baselines.PPO
        total_timesteps = 10000
        ext_name = ""

        # Run
        scripts_baselines.learn(game, losses, model_type, total_timesteps, ext_name)
        game = scripts_baselines.predict(game, losses, model_type, total_timesteps, ext_name)
        util.write_results(game)

    # ---------------------------------------------------------------

    if True:
        # Ray configuration
        env = scripts_ray.shared_parameter_env(game)
        register_env("pug", lambda _: env)
        ray_model_type = scripts_ray.RayModel.PPO
        iterations = 10

        # Run
        # Zet local_mode=True om te debuggen
        ray.init(local_mode=True, logging_level=logging.DEBUG)
        checkpoint = scripts_ray.learn(game, losses, ray_model_type, iterations)
        scripts_ray.predict(game, env, ray_model_type, checkpoint)
        util.write_results(game)
        ray.shutdown()

    # ---------------------------------------------------------------


if __name__ == '__main__':
    run()
