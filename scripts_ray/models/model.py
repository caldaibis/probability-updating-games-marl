from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional

import ray
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import Trainable, register_env

import probability_updating as pu


class Model(ABC):
    game: pu.Game
    env: ParallelPettingZooEnv
    model_type: Type[Trainable]
    name: str

    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], model_type: Type[Trainable], ext_name: Optional[str] = ''):
        self.game = game
        self.env = self._create_env(game)
        self.model_type = model_type
        self.name = f"g={game.name()}_c={losses[pu.cont()].name}_q={losses[pu.quiz()].name}_e={ext_name}"

        register_env("pug", lambda _: self.env)

    @classmethod
    @abstractmethod
    def get_local_dir(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def _create_model_config(cls) -> dict:
        return {
            "env": "pug",
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 10,
            "num_envs_per_worker": 64,
            "framework": "tf",  # "torch"
            "evaluation_interval": 2,
            "evaluation_num_episodes": 1,
            "evaluation_config": {
                "explore": False
            },
        }

    @abstractmethod
    def _create_tune_config(self, iterations: int) -> dict:
        return {
            "name": self.name,
            "config": self._create_model_config(),
            "stop": {"training_iteration": iterations},
            "checkpoint_freq": 2,
            "checkpoint_at_end": True,
            "verbose": 1,
            "local_dir": self.get_local_dir(),
        }

    @classmethod
    @abstractmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        pass

    def learn(self, iterations: int) -> str:
        analysis = ray.tune.run(self.model_type, **self._create_tune_config(iterations))
        return analysis.get_last_checkpoint()

    def predict(self, model_type: Type[Trainable], checkpoint: str):
        model = model_type(config=self._create_model_config())
        model.restore(checkpoint)

        obs = self.env.reset()
        actions = {agent: model.compute_single_action(obs[agent], unsquash_action=True, explore=False) for agent in pu.agents()}

        obs, rewards, dones, infos = self.env.step(actions)
