from __future__ import annotations

from typing import Optional, Type

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

import src.probability_updating as pu
import src.probability_updating.games as games

algos = ['ppo', 'a2c', 'ddpg', 'td3', 'sac', 'impala', 'marwil']


def plot(new: bool, eval: bool, game_type: Type[games.Game], loss: str, interaction_type: str, optimum: float, bottom: Optional[float] = None):
    plt.figure()

    # Load datasets
    data = {}
    for algo in algos:
        try:
            if new:
                data[algo] = pd.read_csv(f'data_new/{game_type.name()}/{loss}/{interaction_type}/{algo}.csv')
            else:
                data[algo] = pd.read_csv(f'data/{loss}/{game_type.name()}/{interaction_type}/{algo}.csv')
        except FileNotFoundError as e:
            pass

    # Plot datasets
    lines = []
    for algo in data:
        if new and eval:
            lines.append(plt.plot(data[algo]['time_total_s'], data[algo]['universal_reward_eval_mean'], label=algo.upper())[0])
        elif new:
            lines.append(plt.plot(data[algo]['time_total_s'], data[algo]['universal_reward_mean'], label=algo.upper())[0])
        else:
            lines.append(plt.plot(data[algo]['time_total_s'], data[algo]['surrogate_reward_mean'], label=algo.upper())[0])

    # Plot miscellaneous
    plt.axhspan(optimum, optimum + 0.004, alpha=0.5, zorder=1, color='gold')
    lines.insert(0, Line2D([0], [0], label='[Nash equilibrium]', color='gold'))

    lower_anchor = 0.0 if game_type.name() == pu.games.MontyHall.name() else 0.07
    plt.legend(frameon=False, handles=lines, loc='lower right', ncol=3, bbox_to_anchor=(1.0, lower_anchor))

    # Plot configurations
    plt.xlim(0, 60)
    if bottom:
        plt.ylim(bottom=bottom)
    plt.title(f"{interaction_type.capitalize()} {game_type.name()} with {loss} loss")
    plt.xlabel("Total time in seconds")
    if new and eval:
        plt.ylabel("Universal reward mean (eval)")
    elif new:
        plt.ylabel("Universal reward mean")
    else:
        plt.ylabel("Surrogate reward mean")

    if new and eval:
        plt.savefig(f'figures_new_eval/{game_type.name()}_{loss}_{interaction_type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)
    elif new:
        plt.savefig(f'figures_new/{game_type.name()}_{loss}_{interaction_type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig(f'figures/{game_type.name()}_{loss}_{interaction_type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)


def show_all(new: bool = True, eval: bool = False):
    # Set style
    sns.set()
    
    # Plot figures
    plot(new, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', optimum=-0.66667)
    plot(new, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='cooperative', optimum=-0.66667)

    plot(new, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type='zero-sum', optimum=-0.888)
    plot(new, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type='cooperative', optimum=-0.666)

    plot(new, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type='zero-sum', optimum=-1.273, bottom=-2.5)
    plot(new, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type='cooperative', optimum=-0.924, bottom=-2.5)

    # plot(new, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', optimum=-1.333)
    # plot(new, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='cooperative', optimum=-1.333)
    #
    # plot(new, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type='zero-sum', optimum=-1.444)
    # plot(new, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type='cooperative', optimum=-1.333)
    #
    # plot(new, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type='zero-sum', optimum=-2.659)
    # plot(new, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type='cooperative', optimum=-2.197)

    # Present plot(s)
    plt.show()


if __name__ == '__main__':
    show_all(new=True, eval=True)
