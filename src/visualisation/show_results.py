from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

import probability_updating as pu


loss_names = {
    pu.Loss.zero_one().name: 'randomised 0-1',
    pu.Loss.logarithmic().name: 'logarithmic',
    pu.Loss.brier().name: 'Brier'
}

game_names = {
    pu.games.MontyHall.name(): 'Monty Hall',
    pu.games.FairDie.name(): 'Fair Die'
}

algos = ['ppo', 'a2c', 'ddpg', 'td3', 'sac']


def plot(game, loss, type, optimum, bottom = None):
    plt.figure()

    # Load datasets
    data = {}
    for algo in algos:
        try:
            data[algo] = pd.read_csv(f'data/{loss}/{game}/{type}/{algo}.csv')[['time_total_s', 'surrogate_reward_mean']]
        except FileNotFoundError as e:
            pass

    # Plot datasets
    lines = []
    for algo in data:
        lines.append(plt.plot(data[algo]['time_total_s'], data[algo]['surrogate_reward_mean'], label=algo.upper())[0])

    # Plot miscellaneous
    plt.axhspan(optimum, optimum + 0.004, alpha=0.5, zorder=1, color='gold')
    lines.insert(0, Line2D([0], [0], label='[Nash equilibrium]', color='gold'))

    lower_anchor = 0.0 if game == pu.games.MontyHall.name() else 0.07
    plt.legend(frameon=False, handles=lines, loc='lower right', ncol=3, bbox_to_anchor=(1.0, lower_anchor))

    # Plot configurations
    plt.xlim(0, 60)
    if bottom:
        plt.ylim(bottom=bottom)
    # plt.title(f"{type.capitalize()} {game_names[game]} with {loss_names[loss]} loss")
    plt.xlabel("Total time in seconds")
    plt.ylabel("Surrogate reward mean")

    plt.savefig(f'figures/{game}_{loss}_{type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)


def show_results():
    # Set style
    sns.set()

    # Plot figures
    plot(game=pu.games.MontyHall.name(), loss=pu.Loss.zero_one().name, type='zero-sum', optimum=-0.66667)
    plot(game=pu.games.MontyHall.name(), loss=pu.Loss.zero_one().name, type='cooperative', optimum=-0.66667)

    plot(game=pu.games.MontyHall.name(), loss=pu.Loss.logarithmic().name, type='zero-sum', optimum=-1.273, bottom=-2.5)
    plot(game=pu.games.MontyHall.name(), loss=pu.Loss.logarithmic().name, type='cooperative', optimum=-0.924, bottom=-2.5)

    plot(game=pu.games.MontyHall.name(), loss=pu.Loss.brier().name, type='zero-sum', optimum=-0.888)
    plot(game=pu.games.MontyHall.name(), loss=pu.Loss.brier().name, type='cooperative', optimum=-0.666)

    plot(game=pu.games.FairDie.name(), loss=pu.Loss.zero_one().name, type='zero-sum', optimum=-1.333)
    plot(game=pu.games.FairDie.name(), loss=pu.Loss.zero_one().name, type='cooperative', optimum=-1.333)

    plot(game=pu.games.FairDie.name(), loss=pu.Loss.logarithmic().name, type='zero-sum', optimum=-2.659)
    plot(game=pu.games.FairDie.name(), loss=pu.Loss.logarithmic().name, type='cooperative', optimum=-2.197)

    plot(game=pu.games.FairDie.name(), loss=pu.Loss.brier().name, type='zero-sum', optimum=-1.444)
    plot(game=pu.games.FairDie.name(), loss=pu.Loss.brier().name, type='cooperative', optimum=-1.333)

    # Present plot(s)
    # plt.show()


if __name__ == '__main__':
    show_results()
