from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Tuple

import ray
from ray.rllib import MultiAgentEnv
from ray.rllib.agents import Trainer
from ray.rllib.models import ModelCatalog
from ray.tune import Trainable, register_env, sample_from, ExperimentAnalysis
from ray.tune.stopper import CombinedStopper, ExperimentPlateauStopper, Stopper
from ray.tune.progress_reporter import CLIReporter

import probability_updating as pu
from src.scripts_ray.stoppers import ConjunctiveStopper, TotalTimeStopper


class Model(ABC):
    game: pu.Game
    trainer_type: Type[Trainer]
    env: MultiAgentEnv
    hyper_param: Dict
    name: str
    metric: str
    reporter: CLIReporter

    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainer], hyper_param: Dict, min_total_time_s: int, max_total_time_s: int):
        self.game = game
        self.trainer_type = trainer_type
        self.env = self._create_env(game)
        self.hyper_param = hyper_param
        self.min_total_time_s = min_total_time_s
        self.max_total_time_s = max_total_time_s
        self.reporter = CLIReporter(max_report_frequency=10)

        self.name = f"{game.name()}_{pu.Agent.Cont}={losses[pu.Agent.Cont].name}_{pu.Agent.Host}={losses[pu.Agent.Host].name}"

        register_env("pug", lambda _: self.env)

    @abstractmethod
    def get_local_dir(self) -> str:
        pass

    @abstractmethod
    def _create_model_config(self) -> dict:
        return {
            **self.hyper_param,
            "env": "pug",
            "batch_mode": "truncate_episodes",
            "num_gpus": 0,
            "num_cpus_for_driver": 1,
            "num_cpus_per_worker": 1,
            "framework": "torch",
            "evaluation_interval": 5,
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
            "stop": CombinedStopper(ConjunctiveStopper(ExperimentPlateauStopper(self.metric, mode="max", top=10, std=0.0005), TotalTimeStopper(total_time_s=self.min_total_time_s)), TotalTimeStopper(total_time_s=self.max_total_time_s)),
            "checkpoint_freq": 5,
            "checkpoint_at_end": True,
            "local_dir": self.get_local_dir(),
            "verbose": 1,
            "metric": self.metric,
            "mode": "max",
            "progress_reporter": self.reporter,
        }

    @classmethod
    @abstractmethod
    def _create_env(cls, game: pu.Game) -> MultiAgentEnv:
        pass

    def learn(self) -> ExperimentAnalysis:
        return ray.tune.run(self.trainer_type, **self._create_tune_config())

    def save_to_results(self, analysis):
        import shutil
        import os
        from pathlib import Path

        loss = self.game.loss[pu.Agent.Cont].name
        same = self.game.loss[pu.Agent.Cont].name == self.game.loss[pu.Agent.Host].name
        type_t = 'cooperative' if same else 'zero-sum'

        algo = self.trainer_type._name

        original = Path(f'{analysis.best_logdir}/progress.csv')
        destination_dir = Path(f'../visualisation/data/{loss}/{self.game.name()}/{type_t}/')

        if os.path.isfile(Path(destination_dir / f'{algo.lower()}.csv')):
            i = 1
            while os.path.isfile(destination_dir / f'{algo.lower()}{str(i)}.csv'):
                i += 1
            destination = destination_dir / f'{algo.lower()}{str(i)}.csv'
        else:
            destination = destination_dir / f'{algo.lower()}.csv'

        shutil.copy(original, destination)

    def load(self) -> Optional[str]:
        """Safely loads an existing checkpoint. If none exists, returns None"""
        try:
            analysis = ExperimentAnalysis(f"{self.get_local_dir()}/{self.name}", default_metric=self.metric, default_mode="max")
            return analysis.best_checkpoint
        except Exception as e:
            return None

    @abstractmethod
    def predict(self, checkpoint: str):
        pass
