from __future__ import annotations

import os
from abc import ABC, abstractmethod
import random
from typing import Dict, Optional, Type, Tuple

import ray
from ray.rllib.agents import Trainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import Trainable, register_env, sample_from, ExperimentAnalysis
from ray.tune.result import EPISODE_REWARD_MEAN
from ray.tune.stopper import CombinedStopper, TimeoutStopper, ExperimentPlateauStopper, MaximumIterationStopper

import probability_updating as pu

from ray.tune.progress_reporter import CLIReporter


class Model(ABC):
    game: pu.Game
    trainer_type: Type[Trainer]
    env: ParallelPettingZooEnv
    name: str

    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainer] = PPOTrainer, ext_name: str = ''):
        self.game = game
        self.trainer_type = trainer_type
        self.env = self._create_env(game)
        self.name = f"g={game.name()}_c={losses[pu.Agent.Cont].name}_q={losses[pu.Agent.Host].name}{'_t='+trainer_type.__name__ if ext_name else ''}{'_e='+ext_name if ext_name else ''}"

        register_env("pug", lambda _: self.env)

    @abstractmethod
    def get_local_dir(self) -> str:
        pass

    @abstractmethod
    def _create_model_config(self) -> dict:
        return {
            "env": "pug",
            "num_gpus": 0,  # int(os.environ.get("RLLIB_NUM_GPUS", "0"))
            "num_cpus_for_driver": 1,
            "num_cpus_per_worker": 1,
            "framework": "torch",  # "tf"
            "evaluation_interval": 1,
            "evaluation_num_episodes": 1,
            "evaluation_config": {
                "explore": False
            },
        }

    @abstractmethod
    def _create_tune_config(self, timeout_seconds: float, hyper_param: Dict) -> dict:
        return {
            "name": self.name,
            "config": {**self._create_model_config(), **hyper_param},
            "stop": ExperimentPlateauStopper(EPISODE_REWARD_MEAN, mode="max", top=10, std=0.001),  # {"time_total_s": timeout_seconds},  # training_iteration, timesteps_total,   # TimeoutStopper(timeout_seconds),  # CombinedStopper(TimeoutStopper(timeout_seconds), ExperimentPlateauStopper(EPISODE_REWARD_MEAN, mode="max")),
            "checkpoint_freq": 1,
            "checkpoint_at_end": True,
            "local_dir": self.get_local_dir(),
            "verbose": 1,
            "metric": "episode_reward_mean",
            "mode": "max",
            "progress_reporter": CLIReporter(),
        }

    @classmethod
    @abstractmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        pass

    def learn(self, timeout_seconds: float, hyper_param: Dict) -> str:
        analysis = ray.tune.run(self.trainer_type, **self._create_tune_config(timeout_seconds, hyper_param))
        # best_trial = analysis.best_trial  # Get best trial
        # best_config = analysis.best_config  # Get best trial's hyperparameters
        # best_logdir = analysis.best_logdir  # Get best trial's logdir
        # best_result = analysis.best_result  # Get best trial's last results
        # best_result_df = analysis.best_result_df  # Get best result as pandas dataframe
        return analysis.best_checkpoint  # Get best trial's best checkpoint

    def load(self) -> Optional[str]:
        """Safely loads an existing checkpoint. If none exists, returns None"""
        try:
            analysis = ExperimentAnalysis(f"{self.get_local_dir()}/{self.name}", default_metric=EPISODE_REWARD_MEAN, default_mode="max")
            return analysis.best_checkpoint
        except Exception:
            return None

    @abstractmethod
    def predict(self, checkpoint: str):
        pass