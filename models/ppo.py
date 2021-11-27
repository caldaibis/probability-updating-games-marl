from __future__ import annotations

from stable_baselines3 import PPO as ppo
from stable_baselines3.ppo import MlpPolicy

import supersuit as ss
import models


class PPO(models.Model):
    @classmethod
    def _apply(cls, env: ss.vector.MarkovVectorEnv):
        return ppo(MlpPolicy, env, verbose=1)  # n_steps=10, batch_size=20, learning_rate=0.1

    @classmethod
    def _load(cls, model_path: str):
        return ppo.load(model_path)

    @staticmethod
    def name() -> str:
        return "PPO"
