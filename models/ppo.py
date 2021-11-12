from __future__ import annotations

from stable_baselines3 import PPO as ppo
from stable_baselines3.ppo import MlpPolicy

import supersuit as ss
import models


class PPO(models.Model):
    @classmethod
    def _apply(cls, env: ss.vector.MarkovVectorEnv):
        return ppo(MlpPolicy, env, verbose=3)

    @classmethod
    def _load(cls, model_path: str):
        return ppo.load(model_path)

    @staticmethod
    def name() -> str:
        return "PPO"
