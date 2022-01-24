from __future__ import annotations

from typing import List

from ray.tune.trial import Trial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def show_figure(trials: List[Trial], max_total_time_s):
    sns.set()

    plt.figure()

    data = {}
    for t in trials:
        data[t] = pd.read_csv(f'{t.logdir}/progress.csv')
        
    for key in data:
        plt.plot(data[key]['time_total_s'], data[key]['universal_reward_mean'], label=key.experiment_tag)
        plt.plot(data[key]['time_total_s'], data[key]['universal_reward_eval_mean'], label=key.experiment_tag + ' (eval)')

    plt.legend(frameon=False, loc='lower right', ncol=1)

    plt.xlim(0, max(60, max_total_time_s))
    plt.xlabel("Total time in seconds")
    plt.ylabel("Reward mean")

    plt.figure()

    for key in data:
        plt.plot(data[key]['time_total_s'], data[key]['rcar_rmse_mean'], label=key.experiment_tag + ' RCAR RMSE')
        plt.plot(data[key]['time_total_s'], data[key]['rcar_rmse_eval_mean'], label=key.experiment_tag + ' RCAR RMSE (eval)')

    plt.legend(frameon=False, loc='lower right', ncol=1)

    plt.xlim(0, max(60, max_total_time_s))
    plt.xlabel("Total time in seconds")
    plt.ylabel("RMSE")

    plt.show()
