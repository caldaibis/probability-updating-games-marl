from __future__ import annotations

from typing import Dict, Type, List

import ray
from matplotlib import pyplot as plt
from ray.rllib import MultiAgentEnv
from ray.rllib.agents import Trainer
from ray.rllib.policy.policy import PolicySpec
from ray.tune import Trainable, register_env, ExperimentAnalysis
from ray.tune.stopper import CombinedStopper, ExperimentPlateauStopper
from ray.tune.progress_reporter import CLIReporter
from ray.tune.trial import Trial

import src.lib_pu as pu
import src.lib_marl as marl

import shutil
import os
from pathlib import Path

import src.lib_vis as vis


class ModelWrapper:
    game: pu.Game
    trainer_type: Type[Trainer]
    env: MultiAgentEnv
    custom_config: Dict
    name: str
    exp_config: dict
    metric: str
    reporter: CLIReporter

    def __init__(self, experiment_name, game: pu.Game, losses: Dict[pu.Agent, str], trainer_type: Type[Trainer], custom_config: Dict):
        self.experiment_name = experiment_name
        self.game = game
        self.trainer_type = trainer_type
        self.env = self._create_env(game)
        self.custom_config = custom_config
        self.reporter = CLIReporter(max_report_frequency=5)

        self.name = f"{pu.CONT}={losses[pu.CONT]}_{pu.HOST}={losses[pu.HOST]}_t={custom_config['max_total_time_s']}"
        self.exp_config = {
            'game': self.game.name(),
            pu.CONT: losses[pu.CONT],
            pu.HOST: losses[pu.HOST],
            't': custom_config['max_total_time_s'],
            'save_figures': custom_config['save_figures'],
        }
        
        register_env("pug", lambda _: self.env)

        self.reporter.add_metric_column(marl.REWARD_CONT_EVAL)
        self.reporter.add_metric_column(marl.REWARD_HOST_EVAL)

        self.reporter.add_metric_column(marl.RCAR_DIST_EVAL)
        self.reporter.add_metric_column(marl.EXP_ENTROPY_EVAL)

        self.reporter.add_metric_column(marl.REWARD_CONT)
        self.reporter.add_metric_column(marl.REWARD_HOST)

        self.reporter.add_metric_column(marl.RCAR_DIST)
        self.reporter.add_metric_column(marl.EXP_ENTROPY)

        self.metric = "universal_reward_mean"

    def get_local_dir(self) -> str:
        return f"output_ray/{self.trainer_type.__name__}/{self.game.name()}/"

    def __call__(self, learn: bool, show_figure: bool, show_eval: bool, save_progress: bool) -> None:
        if learn:
            analysis = ray.tune.run(self.trainer_type, **self._create_tune_config())
            if save_progress:
                self._save_progress(analysis)
        
        analysis = ExperimentAnalysis(self._get_experiment_paths(), default_metric=self.metric, default_mode="max")
        trials = [trial for trial in analysis.trials if trial.checkpoint.value]
        
        trial_action_data = self.predict_all_trials(trials)
        if show_figure or self.exp_config['save_figures']:
            vis.init()
            vis.show_strategy_figures(self.exp_config, trial_action_data, self.game.outcomes, self.game.messages)
            
            if show_eval:
                metrics = {
                    'performance': [marl.REWARD_CONT_EVAL, marl.REWARD_HOST_EVAL, marl.EXP_ENTROPY_EVAL],
                    'rcar':  [marl.RCAR_DIST_EVAL],
                }
            else:
                metrics = {
                    'performance': [marl.REWARD_CONT, marl.REWARD_HOST, marl.EXP_ENTROPY],
                    'rcar':  [marl.RCAR_DIST],
                }
            vis.show_aggregated_metric(self.exp_config, trials, marl.RCAR_DIST_EVAL)
            vis.show_multiple_aggregated_metrics(self.exp_config, trials, metrics['performance'], y_label='Expected loss')
            
            for trial in trials:
                vis.show_trial_data(trial, self.game.outcomes, self.game.messages, self.exp_config)
        
        if show_figure:
            plt.show()

    def _get_experiment_paths(self) -> List[str]:
        return [f'{self.get_local_dir()}/{self.name}/{f}' for f in os.listdir(f'{self.get_local_dir()}/{self.name}/') if f.startswith("experiment_state")]

    """Predict by all trials of the experiment"""
    def predict_all_trials(self, trials: List[Trial]) -> Dict[str, List]:
        trainer = self.trainer_type(config=self._create_model_config())
        
        actions_all = {'trial_ids': [], pu.CONT: [], pu.HOST: [], 'host_reverse': []}
        for trial in trials:
            trainer.restore(trial.checkpoint.value)

            obs = self.env.reset()
            actions = {
                agent: trainer.compute_single_action(obs[agent], explore=False, policy_id=agent)
                for agent in pu.AGENTS
            }
            obs, rewards, dones, infos = self.env.step(actions)
            
            print(self.game)
            actions_all['trial_ids'].append(trial.trial_id)
            actions_all[pu.CONT].append(self.game.action[pu.CONT])
            actions_all[pu.HOST].append(self.game.action[pu.HOST])
            actions_all['host_reverse'].append(self.game.host_reverse)
        
        return actions_all

    def _create_model_config(self) -> dict:
        return {
            **self.custom_config['model_config'],
            "env": "pug",
            "batch_mode": "truncate_episodes",
            # "num_gpus": 0,
            # "num_cpus_for_driver": 1,
            # "num_cpus_per_worker": 1,
            "framework": "torch",
            "evaluation_interval": 1,
            "evaluation_num_episodes": 1,
            "evaluation_config": {
                "explore": False
            },
            "multiagent": {
                "policies": {
                    pu.CONT: PolicySpec(
                        None,
                        self.env.observation_spaces[pu.CONT],
                        self.env.action_spaces[pu.CONT],
                        None
                    ),
                    pu.HOST: PolicySpec(
                        None,
                        self.env.observation_spaces[pu.HOST],
                        self.env.action_spaces[pu.HOST],
                        None
                    ),
                },
                "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
            },
            "callbacks": marl.CustomMetricCallbacks,
        }

    def _create_tune_config(self) -> dict:
        return {
            **self.custom_config['tune_config'],
            "name": self.name,
            "config": self._create_model_config(),
            "stop": CombinedStopper(marl.ConjunctiveStopper(ExperimentPlateauStopper(self.metric, mode="max", top=10, std=0.0001), marl.TotalTimeStopper(total_time_s=self.custom_config['min_total_time_s'])), marl.TotalTimeStopper(total_time_s=self.custom_config['max_total_time_s'])),
            "checkpoint_at_end": True,
            "local_dir": self.get_local_dir(),
            "verbose": 1,
            "metric": self.metric,
            "mode": "max",
            "progress_reporter": self.reporter,
        }

    @staticmethod
    def _create_env(game: pu.Game) -> MultiAgentEnv:
        env = pu.ProbabilityUpdatingEnv(game)
        return pu.ProbabilityUpdatingEnvWrapper(env)

    def _save_progress(self, analysis: ExperimentAnalysis):
        loss = self.game.loss_names[pu.CONT]
        same = self.game.loss_names[pu.CONT] == self.game.loss_names[pu.HOST]
        interaction_type = pu.COOPERATIVE if same else pu.ZERO_SUM

        algo = self.trainer_type.__name__

        original = Path(f'{analysis.best_logdir}/progress.csv')
        destination_dir = Path(f'../lib_vis/data/{self.experiment_name}/{self.game.name()}_{loss}_{interaction_type}_{algo.lower()}')

        destination = Path(f'{destination_dir}.csv')
        if os.path.isfile(destination):
            i = 1
            while os.path.isfile(Path(f'{destination_dir}{i}.csv')):
                i += 1
            destination = Path(f'{destination_dir}{i}.csv')

        shutil.copy(original, destination)
