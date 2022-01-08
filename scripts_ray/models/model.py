from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
import random
from typing import Dict, Optional, Type, Tuple

import ray
from ray.rllib.agents import Trainer
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import Trainable, register_env, sample_from, ExperimentAnalysis
from ray.tune.result import EPISODE_REWARD_MEAN
from ray.tune.stopper import CombinedStopper, ExperimentPlateauStopper, MaximumIterationStopper, Stopper

import probability_updating as pu

from ray.tune.progress_reporter import CLIReporter


class Model(ABC):
    game: pu.Game
    trainer_type: Type[Trainer]
    env: ParallelPettingZooEnv
    hyper_param: Dict
    name: str

    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainer], hyper_param: Dict, min_total_time_s: int, max_total_time_s: int):
        self.game = game
        self.trainer_type = trainer_type
        self.env = self._create_env(game)
        self.hyper_param = hyper_param
        self.min_total_time_s = min_total_time_s
        self.max_total_time_s = max_total_time_s

        self.name = f"g={game.name()}_c={losses[pu.Agent.Cont].name}_q={losses[pu.Agent.Host].name}"

        register_env("pug", lambda _: self.env)

    @abstractmethod
    def get_local_dir(self) -> str:
        pass

    @abstractmethod
    def _create_model_config(self) -> dict:
        return {
            **self.hyper_param,
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
    def _create_tune_config(self) -> dict:
        return {
            "name": self.name,
            "config": self._create_model_config(),
            # "stop": ExperimentPlateauStopper(EPISODE_REWARD_MEAN, mode="max", top=10, std=0.001),  # {"time_total_s": timeout_seconds},  # training_iteration, timesteps_total,   # TimeoutStopper(timeout_seconds),  # CombinedStopper(TimeoutStopper(timeout_seconds), ExperimentPlateauStopper(EPISODE_REWARD_MEAN, mode="max")),
            # "stop": {"time_total_s": timeout_seconds},
            "stop": CombinedStopper(ConjunctiveStopper(ExperimentPlateauStopper(EPISODE_REWARD_MEAN, mode="max", top=10, std=0.001), TotalTimeStopper(total_time_s=self.min_total_time_s)), TotalTimeStopper(total_time_s=self.max_total_time_s)),
            "checkpoint_freq": 1,
            "checkpoint_at_end": True,
            "local_dir": self.get_local_dir(),
            "verbose": 3,
            "metric": "episode_reward_mean",
            "mode": "max",
            "progress_reporter": CLIReporter(),
        }

    @classmethod
    @abstractmethod
    def _create_env(cls, game: pu.Game) -> ParallelPettingZooEnv:
        pass

    def learn(self) -> str:
        analysis = ray.tune.run(self.trainer_type, **self._create_tune_config())
        # best_trial = analysis.best_trial  # Get best trial
        # best_config = analysis.best_config  # Get best trial's hyperparameters
        # best_logdir = analysis.best_logdir  # Get best trial's logdir
        # best_result = analysis.best_result  # Get best trial's last results
        # best_result_df = analysis.best_result_df  # Get best result as pandas dataframe
        # return analysis.best_checkpoint  # Get best trial's best checkpoint
        return analysis.get_last_checkpoint()

    def load(self) -> Optional[str]:
        """Safely loads an existing checkpoint. If none exists, returns None"""
        try:
            analysis = ExperimentAnalysis(f"{self.get_local_dir()}/{self.name}", default_metric=EPISODE_REWARD_MEAN, default_mode="max")
            return analysis.best_checkpoint
        except Exception as e:
            return None

    @abstractmethod
    def predict(self, checkpoint: str):
        pass


class ConjunctiveStopper(Stopper):
    """Combine several stoppers via 'AND'."""

    def __init__(self, *stoppers: Stopper):
        self._stoppers = stoppers

    def __call__(self, trial_id, result):
        return all(s(trial_id, result) for s in self._stoppers)

    def stop_all(self):
        return all(s.stop_all() for s in self._stoppers)


class TotalTimeStopper(Stopper):
    def __init__(self, total_time_s: Optional[int] = None):
        self._total_time_s = total_time_s

    def __call__(self, trial_id, result):
        return result["time_total_s"] > self._total_time_s

    def stop_all(self):
        return False
