from __future__ import annotations

from typing import List

from ray.tune.trial import Trial

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def direct(trials: List[Trial]):
    sns.set()

    plt.figure()

    for t in trials:
        data = pd.read_csv(f'{t.logdir}/progress.csv')[['time_total_s', 'universal_reward_mean']]
        plt.plot(data['time_total_s'], data['universal_reward_mean'], label=t.experiment_tag)

    plt.legend(frameon=False, loc='lower right', ncol=1)

    plt.xlim(0, 60)
    plt.xlabel("Total time in seconds")
    plt.ylabel("Reward mean")

    plt.show()