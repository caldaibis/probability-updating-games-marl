from __future__ import annotations

import logging

import src.probability_updating as pu
import src.learning as learning
import src.util as util
import src.scripts_baselines as scripts_baselines

import ray


def run():
    # Essential configuration
    # _games = pu.Game.__subclasses__()
    _games = [pu.games.MontyHall]
    losses = {
        pu.Agent.Cont: pu.Loss.zero_one(),
        pu.Agent.Host: pu.Loss.zero_one()
    }

    # ---------------------------------------------------------------
    # Loop through varying probability updating games
    for g in _games:
        game = g(losses)

        # ---------------------------------------------------------------

        if False:
            # Manual run configuration
            actions = {
                pu.Agent.Cont: game.cont_optimal_zero_one(),
                pu.Agent.Host: game.host_default()
            }

            # Run
            # util.test(game, actions)
            util.manual_step(game, actions)
            print(game)

        # ---------------------------------------------------------------

        if False:
            # Baseline configuration
            total_timesteps = 10000
            ext_name = ""

            # Run
            model = scripts_baselines.Model(game, losses, total_timesteps, ext_name)
            model.learn()
            model.predict()
            print(game)

        # ---------------------------------------------------------------

        if True:
            # Ray configuration
            # ray.init(local_mode=False, logging_level=logging.INFO, log_to_driver=False)  # Running
            ray.init(local_mode=True, logging_level=logging.DEBUG, log_to_driver=True)  # Debugging
            timeout_seconds = 60
            ray_model = learning.ParameterSharingModel(game, losses)

            # Run
            checkpoint = None
            checkpoint = ray_model.load_and_predict()
            if not checkpoint:
                checkpoint = ray_model.learn(timeout_seconds)
            ray_model.predict(checkpoint)
            print(game)

    # ---------------------------------------------------------------

    ray.shutdown()


if __name__ == '__main__':
    run()
