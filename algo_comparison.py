from __future__ import annotations

import logging

import util

import probability_updating as pu

import ray
import scripts_ray

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.ddpg import DDPGTrainer, TD3Trainer
from ray.rllib.agents.sac import SACTrainer


def run():
    # Essential configuration
    losses = {
        pu.Agent.Cont: pu.Loss.zero_one(),
        pu.Agent.Host: pu.Loss.zero_one()
    }
    game = pu.games.MontyHall(losses)

    if True:
        # Manual configuration
        actions = {
            pu.Agent.Cont: game.cont_optimal_zero_one(),
            pu.Agent.Host: game.host_default()
        }

        # Run
        util.manual_step(game, actions)
        print(game)

    if True:
        # Configuration
        ray.init(local_mode=False, logging_level=logging.INFO)  # Zet local_mode=True om te debuggen
        trainers = [PPOTrainer, A2CTrainer, DDPGTrainer, TD3Trainer, SACTrainer]
        t = PPOTrainer
        timeout_seconds = 900

        print()
        print("NOW USING " + t.__name__)
        print()

        ray_model = scripts_ray.ParameterSharingModel(game, losses, t)

        # Run
        checkpoint = ray_model.learn(timeout_seconds)
        ray_model.predict(checkpoint)
        print(game)

    ray.shutdown()


if __name__ == '__main__':
    run()
