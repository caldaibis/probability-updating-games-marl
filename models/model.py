from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import probability_updating as pu
import probability_updating.games as games

import supersuit as ss


class Model(ABC):
    @classmethod
    def create(cls, game: games.Game):
        return cls._apply(cls._create_learn_env(game))

    @classmethod
    def _create_env(cls, game: games.Game):
        env = pu.probability_updating_env.env(game=game, obs=False)
        env = ss.pad_action_space_v0(env)
        env = ss.agent_indicator_v0(env)
        return ss.pettingzoo_env_to_vec_env_v0(env)

    @classmethod
    def predict(cls, game: games.Game, model_path: str) -> (np.ndarray, np.ndarray, np.ndarray, list):
        env = cls._create_env(game)

        model = cls._load(model_path)

        obs = env.reset()

        actions = {
            0: model.predict(obs[0], deterministic=True)[0],
            1: model.predict(obs[1], deterministic=True)[0]
        }

        obs, rewards, dones, infos = env.step(actions)

        return obs, rewards, dones, infos

    @classmethod
    def _create_learn_env(cls, game: games.Game):
        return ss.concat_vec_envs_v0(cls._create_env(game), 1, num_cpus=1, base_class='stable_baselines3')

    @classmethod
    @abstractmethod
    def _apply(cls, env: ss.vector.MarkovVectorEnv):
        pass

    @classmethod
    @abstractmethod
    def _load(cls, model_path: str):
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass
