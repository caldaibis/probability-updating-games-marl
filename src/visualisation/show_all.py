from __future__ import annotations

from typing import Optional, Type

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

import src.probability_updating as pu
import src.probability_updating.games as games

algos = ['ppo']  # 'a2c', 'ddpg', 'td3', 'sac'


def plot(new: bool, old: bool, no_eval: bool, eval: bool, game_type: Type[games.Game], loss: str, interaction_type: str, optimum: float, bottom: Optional[float] = None, rcar: bool = False):
    plt.figure()

    # Load datasets
    data = {'old': {}, 'new': {}}
    for algo in algos:
        try:
            if new:
                data['new'][algo] = pd.read_csv(f'data_new/{game_type.name()}/{loss}/{interaction_type}/{algo}.csv')
            if old and rcar:
                data['old'][algo] = pd.read_csv(f'data_old_rcar_dist/{loss}/{game_type.name()}/{interaction_type}/{algo}.csv')
            elif old:
                data['old'][algo] = pd.read_csv(f'data/{loss}/{game_type.name()}/{interaction_type}/{algo}.csv')
        except FileNotFoundError as e:
            pass

    # Plot datasets
    lines = []
    for algo in data['old']:
        if rcar and eval:
            lines.append(plt.plot(data['old'][algo]['time_total_s'], data['old'][algo]['rcar_dist_eval_mean'], label=('old '+algo.upper()+' (eval)'))[0])
        elif rcar:
            lines.append(plt.plot(data['old'][algo]['time_total_s'], data['old'][algo]['rcar_dist_mean'], label=('old '+algo.upper()))[0])
        else:
            lines.append(plt.plot(data['old'][algo]['time_total_s'], data['old'][algo]['surrogate_reward_mean'], label=('old '+algo.upper()))[0])
    for algo in data['new']:
        if rcar and eval:
            lines.append(plt.plot(data['new'][algo]['time_total_s'], data['new'][algo]['rcar_dist_eval_mean'], label=('new '+algo.upper()+' (eval)'))[0])
        elif rcar:
            lines.append(plt.plot(data['new'][algo]['time_total_s'], data['new'][algo]['rcar_dist_mean'], label=('new '+algo.upper()))[0])
        elif eval:
            lines.append(plt.plot(data['new'][algo]['time_total_s'], data['new'][algo]['universal_reward_eval_mean'], label=('new '+algo.upper()+' (eval)'))[0])
        if no_eval:
            lines.append(plt.plot(data['new'][algo]['time_total_s'], data['new'][algo]['universal_reward_mean'], label=('new '+algo.upper()))[0])
            
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
    if eval and not no_eval:
        plt.ylabel("Universal reward mean (eval)")
    else:
        plt.ylabel("Universal reward mean")

    if new and old:
        plt.savefig(f'figures_new_vs_old/{game_type.name()}_{loss}_{interaction_type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)
    elif new and no_eval and eval:
        plt.savefig(f'figures_no_eval_vs_eval/{game_type.name()}_{loss}_{interaction_type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)
    elif new and eval:
        plt.savefig(f'figures_new_eval/{game_type.name()}_{loss}_{interaction_type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)
    elif new:
        plt.savefig(f'figures_new/{game_type.name()}_{loss}_{interaction_type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)
    else:
        plt.savefig(f'figures/{game_type.name()}_{loss}_{interaction_type}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)


def show_all(new: bool = True, old: bool = False, no_eval: bool = False, eval: bool = True, rcar: bool = False):
    # Set style
    sns.set()
    
    if rcar:
        # Plot rcar figures
        plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', optimum=-0.66667, rcar=rcar)

        plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type='zero-sum', optimum=-0.888, rcar=rcar)

        plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type='zero-sum', optimum=-1.273, bottom=-2.5, rcar=rcar)

        plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', optimum=-1.333, rcar=rcar)

        plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type='zero-sum', optimum=-1.444, rcar=rcar)

        plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type='zero-sum', optimum=-2.659, rcar=rcar)


    # Plot mean reward figures
    plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', optimum=-0.66667)
    plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='cooperative', optimum=-0.66667)

    plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type='zero-sum', optimum=-0.888)
    plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type='cooperative', optimum=-0.666)

    plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type='zero-sum', optimum=-1.273, bottom=-2.5)
    plot(new, old, no_eval, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type='cooperative', optimum=-0.924, bottom=-2.5)

    plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', optimum=-1.333)
    plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='cooperative', optimum=-1.333)

    plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type='zero-sum', optimum=-1.444)
    plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type='cooperative', optimum=-1.333)

    plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type='zero-sum', optimum=-2.659)
    plot(new, old, no_eval, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type='cooperative', optimum=-2.197)

    # Present plot(s)
    plt.show()


if __name__ == '__main__':
    show_all(new=True, old=True, no_eval=False, eval=True, rcar=True)
