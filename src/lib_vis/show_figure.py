from __future__ import annotations

from typing import List

from ray.tune.trial import Trial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.lib_marl as marl


def _show_figure_for(max_total_time_s, data, metric):
    plt.figure()

    for key in data:
        plt.plot(data[key]['time_total_s'], data[key][metric], label=key.experiment_tag)

    plt.legend(frameon=False, loc='lower right', ncol=2)

    plt.xlim(0, max(60, max_total_time_s))
    plt.xlabel("Total time in seconds")
    plt.ylabel(metric)


def show_figure(trials: List[Trial], max_total_time_s):
    sns.set()

    data = {}
    for t in trials:
        data[t] = pd.read_csv(f'{t.logdir}/progress.csv')

    _show_figure_for(max_total_time_s, data, marl.REWARD_CONT)
    _show_figure_for(max_total_time_s, data, marl.REWARD_CONT_EVAL)
    _show_figure_for(max_total_time_s, data, marl.RCAR_DIST_EVAL)

    plt.show()
