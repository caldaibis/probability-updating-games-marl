from __future__ import annotations

from typing import List, Dict

from ray.tune.trial import Trial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.lib_pu as pu
import src.lib_marl as marl


def show_performance_figure_expectation(title: str, trials: List[Trial], metrics: List[str]):
    sns.set_theme(color_codes=True)
    plt.figure()

    dfs = []
    for t in trials:
        dfs.append(pd.read_csv(f'{t.logdir}/progress.csv'))
    
    # update max coordinate of the x axis
    x_max = max(df.iloc[-1]['time_total_s'] for df in dfs)

    for metric in metrics:
        interpolated_dfs = []
        for orig_df in dfs:
            df = orig_df[['time_total_s', metric]].copy()
            
            # apply sign
            if metric in marl.NEGATIVE_METRICS:
                df[metric] *= -1
            
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
        aggregate_df['min'] = aggregate_df.min(axis=1)
        aggregate_df['max'] = aggregate_df.max(axis=1)
        
        plt.fill_between(
            x=aggregate_df.index.to_series().dt.total_seconds(),
            y1=aggregate_df['min'],
            y2=aggregate_df['max'],
            alpha=0.3
        )
        plt.plot(aggregate_df.index.to_series().dt.total_seconds(), aggregate_df['mean'], label=marl.ALL_METRICS[metric])  # label=key.experiment_tag
        
    plt.legend(frameon=False, loc='lower right', ncol=1)

    plt.xlim(0, max(30, x_max))
    plt.xlabel("Total time in seconds")
    plt.ylabel("Loss")
    plt.title(title)


def show_performance_figure(title: str, trials: List[Trial], metrics: List[str]):
    sns.set_theme(color_codes=True)
    for t in trials:
        plt.figure()
        
        df = pd.read_csv(f'{t.logdir}/progress.csv')
        
        x_max = 0
        for metric in metrics:
            if metric in marl.NEGATIVE_METRICS:
                df[metric] = -1*df[metric]
            
            plt.plot(df['time_total_s'], df[metric], label=marl.ALL_METRICS[metric])  # label=key.experiment_tag
            
            # update max coordinate of the x axis
            x_max = max(x_max, df.iloc[-1]['time_total_s'])
    
        plt.legend(frameon=False, loc='lower right', ncol=1)
    
        plt.xlim(0, max(30, x_max))
        plt.xlabel("Total time in seconds")
        plt.ylabel("Loss")
        plt.title(title)


def show_strategy_figures(actions: Dict[pu.Agent, List[pu.Action]], outcomes: List[pu.Outcome], messages: List[pu.Message]):
    sns.set_theme(color_codes=True)
    _show_cont_figure(actions[pu.CONT], messages)
    _show_host_figure(actions[pu.HOST], outcomes)
    

def _show_cont_figure(actions: List[pu.Action], messages: List[pu.Message]):
    for y in messages:
        if len(y.outcomes) < 2:
            continue
            
        ds = []
        plt.figure()
        
        for action in actions:
            ds.append({str(x): action[x, y] for x in y.outcomes})
        
        df = pd.DataFrame(ds)
        sns.boxplot(data=df, color='tab:blue', width=0.5)
        
        plt.ylim(-0.1, 1.1)
        plt.xlabel(r"$x \in \mathcal{X}$")
        plt.ylabel(r"$Q(x \mid " + f"{y})$")
        plt.title(r"$Q(x \mid " + f"{y})$")


def _show_host_figure(actions: List[pu.Action], outcomes: List[pu.Outcome]):
    for x in outcomes:
        if len(x.messages) < 2:
            continue
            
        ds = []
        plt.figure()
        
        for action in actions:
            ds.append({str(y): action[x, y] for y in x.messages})
        
        df = pd.DataFrame(ds)
        sns.boxplot(data=df, color='tab:orange', width=0.5)
        
        plt.ylim(-0.1, 1.1)
        plt.xlabel(r"$y \in \mathcal{Y}$")
        plt.ylabel(r"$P(y \mid " + f"{x})$")
        plt.title(r"$P(y \mid " + f"{x})$")

