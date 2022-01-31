from __future__ import annotations

from typing import List

from ray.tune.trial import Trial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.lib_marl as marl


def _show_figure_for(df, metrics):
    plt.figure()

    x_max = 0
    for metric in metrics:
        if metric in marl.NEGATIVE_METRICS:
            df[metric] = -1*df[metric]
        
        plt.plot(df['time_total_s'], df[metric], label=marl.ALL_METRICS[metric])  # label=key.experiment_tag
        
        # update max coordinate of the x axis
        x_max = max(x_max, df.iloc[-1]['time_total_s'])

    plt.legend(frameon=False, loc='lower right', ncol=1)

    plt.xlim(0, x_max)
    plt.xlabel("Total time in seconds")
    plt.ylabel("Loss")


def show_figure(trials: List[Trial]):
    sns.set()

    for t in trials:
        df = pd.read_csv(f'{t.logdir}/progress.csv')
        _show_figure_for(df, [marl.REWARD_CONT, marl.REWARD_HOST, marl.EXP_ENTROPY])
        # _show_figure_for(max_total_time_s, df, [marl.REWARD_CONT_EVAL, marl.REWARD_HOST_EVAL, marl.EXP_ENTROPY_EVAL])

    plt.show()
