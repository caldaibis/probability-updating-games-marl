from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import src.probability_updating as pu
import src.probability_updating.games as games

import supersuit as ss

local_dir: Path = Path("../../output_baselines")


class Model:
    game: pu.Game
    checkpoint: str
    total_timesteps: int

    def __init__(self, game: games.Game, losses: Dict[pu.Agent, pu.Loss], total_timesteps: int, ext_name: str = ''):
        self.game = game
        self.checkpoint = self.get_checkpoint_path(game, losses, total_timesteps, ext_name)
        self.total_timesteps = total_timesteps

    def learn(self):
        model = self._create()
        model.learn(total_timesteps=self.total_timesteps)

        model.save(self.checkpoint)

    def predict(self) -> (np.ndarray, np.ndarray, np.ndarray, list):
        env = self._create_env()

        model = PPO.load(self.checkpoint)

        obs = env.reset()

        actions = {
            0: model.predict(obs[0], deterministic=True)[0],
            1: model.predict(obs[1], deterministic=True)[0]
        }

        obs, rewards, dones, infos = env.step(actions)

        return obs, rewards, dones, infos

    @staticmethod
    def get_checkpoint_path(game: games.Game, losses: Dict[pu.Agent, pu.Loss], total_timesteps: int, ext_name: str = ''):
        filename = f"c={losses[pu.Agent.Cont].name}_q={losses[pu.Agent.Host].name}_tt={total_timesteps}_e={ext_name}"

        return f"{local_dir}/{game.name()}/{filename}"

    def _create(self):
        return PPO(MlpPolicy, self._create_learn_env(), verbose=1)  # n_steps=10, batch_size=20, learning_rate=0.1

    def _create_env(self):
        env = pu.ProbabilityUpdatingEnv(self.game)
        env = ss.pad_action_space_v0(env)
        env = ss.agent_indicator_v0(env)
        return ss.pettingzoo_env_to_vec_env_v0(env)

    def _create_learn_env(self):
        return ss.concat_vec_envs_v0(self._create_env(), 1, num_cpus=1, base_class='stable_baselines3')
