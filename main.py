from __future__ import annotations

import logging

import util

import probability_updating as pu
from probability_updating import games

import scripts_baselines

import ray
from ray.tune import register_env
import scripts_ray


from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.ddpg import TD3Trainer, DDPGTrainer
from ray.rllib.agents.pg import PGTrainer


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
        util.test(game_type, losses, actions)

    # ---------------------------------------------------------------

    if False:
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
        iterations = 10
        ray_model = scripts_ray.ParameterSharingModel(game, losses, PPOTrainer)

        # Run
        # Zet local_mode=True om te debuggen
        ray.init(local_mode=True, logging_level=logging.DEBUG)
        checkpoint = ray_model.learn(iterations)
        ray_model.predict(model_type, checkpoint)
        util.write_results(game)
        ray.shutdown()

    # ---------------------------------------------------------------


if __name__ == '__main__':
    run()
