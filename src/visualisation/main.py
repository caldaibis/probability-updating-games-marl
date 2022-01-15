from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run():
    df = pd.read_csv('progress.csv')

    sns.set_theme(style="darkgrid")
    sns.lineplot(x="time_total_s", y="surrogate_reward_mean", data=df)
    plt.show()


if __name__ == '__main__':
    run()

