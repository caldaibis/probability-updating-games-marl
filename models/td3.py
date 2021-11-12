from __future__ import annotations

from stable_baselines3 import TD3 as td3
from stable_baselines3.td3 import MlpPolicy

import supersuit as ss
import models


class TD3(models.Model):
    @classmethod
    def _apply(cls, env: ss.vector.MarkovVectorEnv):
        return td3(MlpPolicy, env, verbose=3)

    @classmethod
    def _load(cls, model_path: str):
        return td3.load(model_path)

    @staticmethod
    def name() -> str:
        return "TD3"
