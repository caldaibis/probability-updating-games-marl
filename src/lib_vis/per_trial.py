from __future__ import annotations

from typing import List

from ray.tune.trial import Trial

import matplotlib.pyplot as plt
import pandas as pd

import src.lib_pu as pu
import src.lib_marl as marl
import src.lib_vis as vis


def _show_trial_metric_over_time(trial_id, df: pd.DataFrame, metric, config, label=None):
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
    base_figure_config = {
        'directory': f'actions_per_trial',
    }
    
    # CONT
    fig, axs = plt.subplots(2, sharex=True)
    for ax in axs.flat:
        ax.label_outer()
    
    plot_configs = []
    plot_configs.append({
        'title': r'Cont: $Q(x \mid y)$',
        'xlabel': 'Time (s)',
        'ylabel': r'$Q(x \mid y)$',
        'ylim': (-0.03, 1.03),
        'legend': True,
    })
    plot_configs.append({
        'xlabel': 'Time (s)',
        'ylabel': 'Expected loss',
        'ylim': vis.get_y_min_max([(marl.REWARD_CONT_EVAL, [df])], config),
        'yticks': vis.get_y_ticks(config, [marl.REWARD_CONT_EVAL]),
        'legend': True,
    })
    
    for y in messages:
        if len(y.outcomes) < 2:
            continue
        
        for x in y.outcomes:
            axs[0].plot(df['time_total_s'], df[f'evaluation/custom_metrics/{pu.CONT}_{y}_{x}_mean'], label=r"$Q("+str(x)+"\mid "+str(y)+")$")
    
    axs[1].plot(df['time_total_s'], df[marl.REWARD_CONT_EVAL], label='Cont expected loss')
    
    figure_config = {
        **base_figure_config,
        'filename': f'{trial_id}_cont.png',
    }
    fig.tight_layout(pad=3.0)
    vis.set_subplot_figure(config, axs, figure_config, plot_configs)
    
    # HOST
    fig, axs = plt.subplots(2, sharex=True)
    for ax in axs.flat:
        ax.label_outer()
    
    plot_configs = []
    plot_configs.append({
        'title': r'Host: $P(y \mid x)$',
        'xlabel': 'Time (s)',
        'ylabel': r'$P(y \mid x)$',
        'ylim': (-0.03, 1.03),
        'legend': True,
    })
    plot_configs.append({
        'xlabel': 'Time (s)',
        'ylabel': 'Expected loss',
        'ylim': vis.get_y_min_max([(marl.REWARD_HOST_EVAL, [df])], config),
        'yticks': vis.get_y_ticks(config, [marl.REWARD_HOST_EVAL]),
        'legend': True,
    })
    
    for x in outcomes:
        if len(x.messages) < 2:
            continue
        
        for y in x.messages:
            axs[0].plot(df['time_total_s'], df[f'evaluation/custom_metrics/{pu.HOST}_{x}_{y}_mean'], label=r"$P("+str(y)+"\mid "+str(x)+")$")
    
    axs[1].plot(df['time_total_s'], df[marl.REWARD_HOST_EVAL], label='Host expected loss', color='tab:orange')
    
    figure_config = {
        **base_figure_config,
        'filename': f'{trial_id}_host.png',
    }
    fig.tight_layout(pad=3.0)
    vis.set_subplot_figure(config, axs, figure_config, plot_configs)


def show_trial_data(trial: Trial, outcomes: List[pu.Outcome], messages: List[pu.Message], config):
    df = pd.read_csv(f'{trial.logdir}/progress.csv')
    
    # RCAR distance
    _show_trial_metric_over_time(trial.trial_id, df, marl.RCAR_DIST_EVAL, config, label=marl.RCAR_DIST_EVAL)
    
    # Cont loss, host loss, entropy
    _show_trial_metrics_over_time(trial.trial_id, df, [marl.REWARD_CONT_EVAL, marl.REWARD_HOST_EVAL, marl.EXP_ENTROPY_EVAL], 'expected_loss', 'Expected loss', config)
    
    # Show learner stats metrics
    _show_alternate_metrics(trial.trial_id, df, config)
    
    # Show evolution of strategies over time
    try:
        _show_trial_actions_over_time(trial.trial_id, df, outcomes, messages, config)
    except KeyError as e:
        pass
