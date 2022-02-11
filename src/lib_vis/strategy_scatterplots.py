from __future__ import annotations

from typing import List, Dict


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.lib_pu as pu
import src.lib_vis as vis


def _show_cont_figure(config, trial_action_data: Dict[str, List], messages: List[pu.Message], is_host_reverse: bool):
    for y in messages:
        if len(y.outcomes) < 2:
            continue
            
        plt.figure()
        ds = {}
        for i in range(len(trial_action_data['trial_ids'])):
            ds[trial_action_data['trial_ids'][i]] = {str(x): trial_action_data[pu.CONT][i][x, y] for x in y.outcomes}
        
        df = pd.DataFrame(ds)
        df = pd.melt(df.reset_index(), id_vars='index', value_vars=trial_action_data['trial_ids'])
        
        sns.stripplot(data=df, x='index', y='value', size=7, hue='variable', dodge=True)
        
        base_plot_config = {
            'directory': '',
            'x_label': r'$x \in \mathcal{X}$',
            'legend': True,
            'y_lim': (-0.1, 1.1),
            'y_ticks': None,
        }
        
        if is_host_reverse:
            plot_config = {
                **base_plot_config,
                'filename': f'host_reverse_{y}.png',
                'title': r'Host reverse: $P(x \mid ' + f'{y})$',
                'y_label': r'$P(x \mid ' + f'{y})$',
            }
        else:
            plot_config = {
                **base_plot_config,
                'filename': f'cont_{y}.png',
                'title': r'Cont: $Q(x \mid ' + f'{y})$',
                'y_label': r'$Q(x \mid ' + f'{y})$',
            }
        
        vis.set_figure(config, plot_config)


def _show_host_figure(config, trial_action_data: Dict[str, List], outcomes: List[pu.Outcome]):
    for x in outcomes:
        if len(x.messages) < 2:
            continue
            
        plt.figure()
        ds = {}
        for i in range(len(trial_action_data['trial_ids'])):
            ds[trial_action_data['trial_ids'][i]] = {str(y): trial_action_data[pu.HOST][i][x, y] for y in x.messages}
        
        df = pd.DataFrame(ds)
        df = pd.melt(df.reset_index(), id_vars='index', value_vars=trial_action_data['trial_ids'])
        
        sns.stripplot(data=df, x='index', y='value', size=7, hue='variable', dodge=True)
        
        plot_config = {
            'directory': '',
            'filename': f'host_{x}.png',
            'title': r'Host: $P(y \mid ' + f'{x})$',
            'x_label': r'$y \in \mathcal{Y}$',
            'y_label': r'$P(y \mid ' + f'{x})$',
            'legend': True,
            'y_lim': (-0.1, 1.1),
            'y_ticks': None,
        }
        vis.set_figure(config, plot_config)


def show_strategy_figures(config, trial_action_data: Dict[str, List], outcomes: List[pu.Outcome], messages: List[pu.Message]):
    _show_cont_figure(config, trial_action_data, messages, False)
    _show_host_figure(config, trial_action_data, outcomes)
    _show_cont_figure(config, trial_action_data, messages, True)
