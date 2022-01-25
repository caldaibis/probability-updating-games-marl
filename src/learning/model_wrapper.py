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

        self.name = f"{game.name()}_{pu.CONT}={losses[pu.CONT]}_{pu.HOST}={losses[pu.HOST]}"

        register_env("pug", lambda _: self.env)

        self.reporter.add_metric_column("universal_reward_mean")
        self.reporter.add_metric_column("universal_reward_eval_mean")
        
        self.reporter.add_metric_column("rcar_dist_mean")
        self.reporter.add_metric_column("rcar_dist_eval_mean")
        
        self.reporter.add_metric_column("reward_cont_mean")
        self.reporter.add_metric_column("reward_cont_eval_mean")
        
        self.reporter.add_metric_column("reward_host_mean")
        self.reporter.add_metric_column("reward_host_eval_mean")
        
        self.metric = "universal_reward_mean"

    def get_local_dir(self) -> str:
        return f"output_ray/{self.trainer_type.__name__}/"

    def learn(self, predict: bool = False, show_figure: bool = False, save_progress: bool = False) -> None:
        analysis = ray.tune.run(self.trainer_type, **self._create_tune_config())

        if predict:
            self.predict(analysis.best_checkpoint)
            
        if save_progress:
            self._save_progress(analysis)
        
        if show_figure:
            visualisation.show_figure(analysis.trials, self.max_total_time_s)

    def load_and_predict(self) -> None:
        """Loads the best existing checkpoint and predicts. If it fails, it will throw an exception."""
        self.predict(ExperimentAnalysis(f"{self.get_local_dir()}/{self.name}", default_metric=self.metric, default_mode="max").best_checkpoint)

    def predict(self, checkpoint: str):
        trainer = self.trainer_type(config=self._create_model_config())
        trainer.restore(checkpoint)

        obs = self.env.reset()
        actions = {
            agent: trainer.compute_single_action(obs[agent], explore=False, policy_id=agent)
            for agent in pu.AGENTS
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
            "evaluation_interval": 1,
            "evaluation_num_episodes": 1,
            "evaluation_config": {
                "explore": False
            },
            "multiagent": {
                "policies": {
                    pu.CONT: PolicySpec(None, self.env.observation_spaces[pu.CONT],
                                                    self.env.action_spaces[pu.CONT], None),
                    pu.HOST: PolicySpec(None, self.env.observation_spaces[pu.HOST],
                                                    self.env.action_spaces[pu.HOST], None),
                },
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
            },
            "callbacks": src.learning.CustomMetricCallbacks,
        }

    def _create_tune_config(self) -> dict:
        return {
            "name": self.name,
            "config": self._create_model_config(),
            "stop": CombinedStopper(src.learning.ConjunctiveStopper(ExperimentPlateauStopper(self.metric, mode="max", top=10, std=0.0001), src.learning.TotalTimeStopper(total_time_s=self.min_total_time_s)), src.learning.TotalTimeStopper(total_time_s=self.max_total_time_s)),
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
        return pu.ProbabilityUpdatingEnvWrapper(env)

    def _save_progress(self, analysis: ExperimentAnalysis):
        loss = self.game.loss_names[pu.CONT]
        same = self.game.loss_names[pu.CONT] == self.game.loss_names[pu.HOST]
        interaction_type = 'cooperative' if same else 'zero-sum'

        algo = self.trainer_type.__name__

        original = Path(f'{analysis.best_logdir}/progress.csv')
        destination_dir = Path(f'../visualisation/data/new/{self.game.name()}/{loss}/{interaction_type}/')

        if os.path.isfile(Path(destination_dir / f'{algo.lower()}.csv')):
            i = 1
            while os.path.isfile(destination_dir / f'{algo.lower()}{str(i)}.csv'):
                i += 1
            destination = destination_dir / f'{algo.lower()}{str(i)}.csv'
        else:
            destination = destination_dir / f'{algo.lower()}.csv'

        shutil.copy(original, destination)
