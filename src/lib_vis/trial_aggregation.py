from __future__ import annotations

from typing import List

from ray.tune.trial import Trial

import matplotlib.pyplot as plt
import pandas as pd

import src.lib_marl as marl
import src.lib_vis as vis


def _aggregate_dfs(dfs: List[pd.DataFrame], metric) -> pd.DataFrame:
    interpolated_dfs = []
    for df in dfs:
        # insert 0.0th second for cleanness
        df.loc[-1] = [0, df.iloc[1][metric]]
        df.index = df.index + 1
        df = df.sort_index()
        
        # interpolate metric values on the 'time_total_s' column
        df['time_total_s'] = pd.to_timedelta(df['time_total_s'], 's')
        df.index = df['time_total_s']
        del df['time_total_s']
        df = df.resample('500ms', origin='start').mean()
        df[metric] = df[metric].interpolate()
        
        interpolated_dfs.append(df)
    
    # cut of excess rows so all row counts are uniform
    min_row_cnt = min(len(df.index) for df in interpolated_dfs)
    
    # create aggregate df containing all other columns -> to easily aggregate the metric values
    aggregate_df = pd.concat([df.head(min_row_cnt)[metric] for df in interpolated_dfs], axis=1)
    aggregate_df['mean'] = aggregate_df.mean(axis=1)
    aggregate_df['std'] = aggregate_df.std(axis=1)
    aggregate_df['time_total_s'] = aggregate_df.index.to_series().dt.total_seconds()
    return aggregate_df


def show_aggregated_metric(config, trials: List[Trial], metric: str):
    plt.figure()
    
    dfs = []
    for t in trials:
        df = pd.read_csv(f'{t.logdir}/progress.csv')[['time_total_s', metric]]
        if metric in marl.NEGATIVE_METRICS:
            df[metric] *= -1
        dfs.append(df)
    
    aggregate_df = _aggregate_dfs(dfs, metric)
    plt.fill_between(
        x=aggregate_df['time_total_s'],
        y1=aggregate_df['mean'] - aggregate_df['std'],
        y2=aggregate_df['mean'] + aggregate_df['std'],
        alpha=0.3
    )
    plt.plot(aggregate_df['time_total_s'], aggregate_df['mean'], label=marl.ALL_METRICS[metric])
    
    plot_config = {
        'directory': '',
        'filename': f'aggregate_{metric}.png',
        'title': marl.ALL_METRICS[metric],
        'x_label': 'Time (s)',
        'y_label': marl.ALL_METRICS[metric],
        'legend': False,
        'y_lim': vis.get_y_min_max([(metric, dfs)], config),
        'y_ticks': vis.get_y_ticks(config, [metric]),
    }
    vis.set_figure(config, plot_config)


def show_multiple_aggregated_metrics(config, trials: List[Trial], metrics: List[str], y_label: str):
    plt.figure()
    
    df_metric_list = []
    for metric in metrics:
        dfs = []
        for t in trials:
            df = pd.read_csv(f'{t.logdir}/progress.csv')[['time_total_s', metric]]
            if metric in marl.NEGATIVE_METRICS:
                df[metric] *= -1
            dfs.append(df)
    
        aggregate_df = _aggregate_dfs(dfs, metric)
        plt.fill_between(
            x=aggregate_df['time_total_s'],
            y1=aggregate_df['mean'] - aggregate_df['std'],
            y2=aggregate_df['mean'] + aggregate_df['std'],
            alpha=0.3
        )
        plt.plot(aggregate_df['time_total_s'], aggregate_df['mean'], label=marl.ALL_METRICS[metric])
        df_metric_list.append((metric, dfs))
    
    plot_config = {
        'directory': '',
        'filename': f'aggregate_{"_".join(metrics)}.png',
        'title': " & ".join([marl.ALL_METRICS[m] for m in metrics]),
        'x_label': 'Time (s)',
        'y_label': y_label,
        'legend': len(metrics) > 1,
        'y_lim': vis.get_y_min_max(df_metric_list, config),
        'y_ticks': vis.get_y_ticks(config, metrics),
    }
    vis.set_figure(config, plot_config)
