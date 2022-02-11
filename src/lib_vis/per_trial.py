from __future__ import annotations

from typing import List

from ray.tune.trial import Trial

import matplotlib.pyplot as plt
import pandas as pd

import src.lib_pu as pu
import src.lib_marl as marl
import src.lib_vis as vis


def _show_trial_metric_over_time(trial_id, df: pd.DataFrame, metric, config, label=None):
    plt.figure()
    plt.plot(df['time_total_s'], df[metric], label=metric if label is None else label)
    
    plot_config = {
        'directory': f'metrics_per_trial/{trial_id}',
        'filename': f'{label}',
        'title': f'{trial_id}_{label}',
        'x_label': 'Time (s)',
        'y_label': metric,
        'legend': False,
        'y_lim': vis.get_y_min_max([(metric, [df])], config),
        'y_ticks': vis.get_y_ticks(config, [metric]),
    }
    vis.set_figure(config, plot_config)


def _show_trial_metrics_over_time(trial_id, df: pd.DataFrame, metrics: List[str], filename: str, y_label: str, config, yticks=None):
    plt.figure()
    
    for metric in metrics:
        if metric in marl.NEGATIVE_METRICS:
            df[metric] = -1*df[metric]
            
        label = marl.ALL_METRICS[metric] if metric in marl.ALL_METRICS else metric
        plt.plot(df['time_total_s'], df[metric], label=label)
    
    plot_config = {
        'directory': f'metrics_per_trial/{trial_id}',
        'filename': filename,
        'title': f'{trial_id}_{y_label}',
        'x_label': 'Time (s)',
        'y_label': y_label,
        'legend': len(metrics) > 1,
        'y_lim': vis.get_y_min_max([(m, [df]) for m in metrics], config),
        'y_ticks': vis.get_y_ticks(config, metrics),
    }
    vis.set_figure(config, plot_config)


def _show_alternate_metrics(trial_id, df: pd.DataFrame, config):
    learners = {
        pu.CONT: f'info/learner/{pu.CONT}/learner_stats/',
        pu.HOST: f'info/learner/{pu.HOST}/learner_stats/',
    }
    
    metrics = [
        'cur_kl_coeff',
        'total_loss',
        'policy_loss',
        'vf_loss',
        'vf_explained_var',
        'kl',
        'entropy',
    ]
    
    for learner in learners:
        for metric in metrics:
            _show_trial_metric_over_time(trial_id, df, learners[learner] + metric, config, label=f'{learner}_{metric}')


def _show_trial_actions_over_time(trial_id, df: pd.DataFrame, outcomes: List[pu.Outcome], messages: List[pu.Message], config):
    base_plot_config = {
        'directory': f'actions_per_trial/{trial_id}',
        'x_label': 'Time (s)',
        'legend': True,
        'y_lim': (-0.1, 1.1),
        'y_ticks': None,
    }
    
    for y in messages:
        if len(y.outcomes) < 2:
            continue
        
        plt.figure()
        for x in y.outcomes:
            plt.plot(df['time_total_s'], df[f'evaluation/custom_metrics/{pu.CONT}_{y}_{x}_mean'], label=r"$Q("+str(x)+"\mid "+str(y)+")$")
            
        plot_config = {
            **base_plot_config,
            'filename': f'cont_{y}.png',
            'title': r'Cont: $Q(x \mid ' + f'{y})$',
            'y_label': r'$Q(x \mid ' + f'{y})$',
        }
        vis.set_figure(config, plot_config)
        
    for x in outcomes:
        if len(x.messages) < 2:
            continue
        
        plt.figure()
        for y in x.messages:
            plt.plot(df['time_total_s'], df[f'evaluation/custom_metrics/{pu.HOST}_{x}_{y}_mean'], label=r"$P("+str(y)+"\mid "+str(x)+")$")
        
        plot_config = {
            **base_plot_config,
            'filename': f'host_{x}.png',
            'title': r'Host: $P(y \mid ' + f'{x})$',
            'y_label': r'$P(y \mid ' + f'{x})$',
        }
        vis.set_figure(config, plot_config)


def show_trial_data(trial: Trial, outcomes: List[pu.Outcome], messages: List[pu.Message], config):
    df = pd.read_csv(f'{trial.logdir}/progress.csv')
    
    # RCAR distance
    _show_trial_metric_over_time(trial.trial_id, df, marl.RCAR_DIST_EVAL, config)
    
    # Cont loss, host loss, entropy
    _show_trial_metrics_over_time(trial.trial_id, df, [marl.REWARD_CONT_EVAL, marl.REWARD_HOST_EVAL, marl.EXP_ENTROPY_EVAL], 'expected_loss', 'Expected loss', config)
    
    # Show learner stats metrics
    _show_alternate_metrics(trial.trial_id, df, config)
    
    # Show evolution of strategies over time
    _show_trial_actions_over_time(trial.trial_id, df, outcomes, messages, config)
