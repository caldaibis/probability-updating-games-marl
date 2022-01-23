from __future__ import annotations

from typing import Dict, Optional, Type

import ray
from ray.rllib import MultiAgentEnv
from ray.rllib.agents import Trainer
from ray.rllib.policy.policy import PolicySpec
from ray.tune import Trainable, register_env, ExperimentAnalysis
from ray.tune.stopper import CombinedStopper, ExperimentPlateauStopper
from ray.tune.progress_reporter import CLIReporter

import src.probability_updating as pu
import src.learning
import supersuit as ss

import shutil
import os
from pathlib import Path

import src.visualisation as visualisation


class ModelWrapper:
    game: pu.Game
    trainer_type: Type[Trainer]
    env: MultiAgentEnv
    hyper_param: Dict
    name: str
    metric: str
    reporter: CLIReporter

    def __init__(self, game: pu.Game, losses: Dict[pu.Agent, str], trainer_type: Type[Trainer], hyper_param: Dict, min_total_time_s: int, max_total_time_s: int):
        self.game = game
        self.trainer_type = trainer_type
        self.env = self._create_env(game)
        self.hyper_param = hyper_param
        self.min_total_time_s = min_total_time_s
        self.max_total_time_s = max_total_time_s
        self.reporter = CLIReporter(max_report_frequency=10)

        self.name = f"{game.name()}_{pu.Agent.Cont}={losses[pu.Agent.Cont]}_{pu.Agent.Host}={losses[pu.Agent.Host]}"

        register_env("pug", lambda _: self.env)

        self.reporter.add_metric_column("surrogate_reward_mean")
        self.reporter.add_metric_column("rcar_rmse")
        self.reporter.add_metric_column("policy_reward_mean_cont")
        self.reporter.add_metric_column("policy_reward_mean_host")
        self.reporter.add_metric_column("policy_reward_mean_min")
        self.reporter.add_metric_column("policy_reward_mean_max")
        
        self.metric = "surrogate_reward_mean"

    def get_local_dir(self) -> str:
        return f"output_ray/{self.trainer_type.__name__}/"

    def learn(self, show_figure: bool = False, save_figure: bool = False) -> ExperimentAnalysis:
        analysis = ray.tune.run(self.trainer_type, **self._create_tune_config())

        if show_figure:
            visualisation.direct(analysis.trials)

        if save_figure:
            self._save_to_results(analysis)

        return analysis

    def safe_load(self) -> Optional[ExperimentAnalysis]:
        """Safely loads an existing checkpoint. If none exists, returns None"""
        try:
            return ExperimentAnalysis(f"{self.get_local_dir()}/{self.name}", default_metric=self.metric, default_mode="max")
        except Exception as e:
            return None

    def predict(self, checkpoint: str):
        trainer = self.trainer_type(config=self._create_model_config())
        trainer.restore(checkpoint)

        obs = self.env.reset()
        actions = {
            agent.value: trainer.compute_single_action(obs[agent.value], explore=False, policy_id=agent.value)
            for agent in pu.Agent
        }
        obs, rewards, dones, infos = self.env.step(actions)
        print(self.game)

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
            "multiagent": {
                "policies": {
                    pu.Agent.Cont.value: PolicySpec(None, self.env.observation_spaces[pu.Agent.Cont.value],
                                                    self.env.action_spaces[pu.Agent.Cont.value], None),
                    pu.Agent.Host.value: PolicySpec(None, self.env.observation_spaces[pu.Agent.Host.value],
                                                    self.env.action_spaces[pu.Agent.Host.value], None),
                },
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
            },
            "callbacks": src.learning.CustomMetricCallbacks,
            "num_workers": 6,
        }

    def _create_tune_config(self) -> dict:
        return {
            "name": self.name,
            "config": self._create_model_config(),
            "stop": CombinedStopper(src.learning.ConjunctiveStopper(ExperimentPlateauStopper(self.metric, mode="max", top=10, std=0.0005), src.learning.TotalTimeStopper(total_time_s=self.min_total_time_s)), src.learning.TotalTimeStopper(total_time_s=self.max_total_time_s)),
            "checkpoint_freq": 5,
            "checkpoint_at_end": True,
            "local_dir": self.get_local_dir(),
            "verbose": 1,
            "metric": self.metric,
            "mode": "max",
            "progress_reporter": self.reporter,
            "num_samples": 1,
        }

    @staticmethod
    def _create_env(game: pu.Game) -> MultiAgentEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        env = ss.agent_indicator_v0(env)
        return pu.ProbabilityUpdatingEnvWrapper(env)

    def _save_to_results(self, analysis: ExperimentAnalysis):
        loss = self.game.loss[pu.Agent.Cont]
        same = self.game.loss[pu.Agent.Cont] == self.game.loss[pu.Agent.Host]
        type_t = 'cooperative' if same else 'zero-sum'

        algo = self.trainer_type.__name__

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