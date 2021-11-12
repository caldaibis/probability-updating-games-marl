from __future__ import annotations

from stable_baselines3 import A2C as a2c
from stable_baselines3.a2c import MlpPolicy

import supersuit as ss
import models as mc


class A2C(mc.Model):
    @classmethod
    def _apply(cls, env: ss.vector.MarkovVectorEnv):
        return a2c(MlpPolicy, env, verbose=3)

    @classmethod
    def _load(cls, model_path: str):
        return a2c.load(model_path)

    @staticmethod
    def name() -> str:
        return "A2C"
