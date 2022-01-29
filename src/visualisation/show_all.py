from __future__ import annotations

from typing import Optional, Type, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

import src.marl_lib as marl
import src.pu_lib as pu
import src.pu_lib.games as games


def plot(algos: List[str], new: bool, old: bool, eval: bool, game_type: Type[games.Game], loss: str, interaction_type: str, optimum: Optional[float] = None, bottom: Optional[float] = None, rcar: bool = False):
    plt.figure()

    # Load datasets
    data = {'old': {}, 'new': {}}
    for algo in algos:
        try:
            if new:
                data['new'][algo] = pd.read_csv(f'data/new/{game_type.name()}/{loss}/{interaction_type}/{algo}.csv')
            if (old and rcar) or (new and old):
                data['old'][algo] = pd.read_csv(f'data/old_rcar_dist/{game_type.name()}/{loss}/{interaction_type}/{algo}.csv')
            elif old:
                data['old'][algo] = pd.read_csv(f'data/old/{loss}/{game_type.name()}/{interaction_type}/{algo}.csv')
        except FileNotFoundError as e:
            pass

    # Plot datasets
    lines = []
    for algo in data['old']:
        if rcar and eval:
            lines.append(plt.plot(data['old'][algo]['time_total_s'], data['old'][algo]['rcar_dist_eval_mean'], label=('old '+algo.upper()))[0])
        elif rcar:
            lines.append(plt.plot(data['old'][algo]['time_total_s'], data['old'][algo]['rcar_dist_mean'], label=('old '+algo.upper()))[0])
        elif new and eval:
            lines.append(plt.plot(data['old'][algo]['time_total_s'], data['old'][algo]['universal_reward_eval_mean'], label=('old '+algo.upper()))[0])
        elif new:
            lines.append(plt.plot(data['old'][algo]['time_total_s'], data['old'][algo]['universal_reward_mean'], label=('old '+algo.upper()))[0])
        else:
            lines.append(plt.plot(data['old'][algo]['time_total_s'], data['old'][algo]['surrogate_reward_mean'], label=('old '+algo.upper()))[0])
            
    for algo in data['new']:
        if rcar and eval:
            lines.append(plt.plot(data['new'][algo]['time_total_s'], data['new'][algo]['rcar_dist_eval_mean'], label=('new '+algo.upper()))[0])
        elif rcar:
            lines.append(plt.plot(data['new'][algo]['time_total_s'], data['new'][algo]['rcar_dist_mean'], label=('new '+algo.upper()))[0])
        elif eval:
            lines.append(plt.plot(data['new'][algo]['time_total_s'], data['new'][algo]['universal_reward_eval_mean'], label=('new '+algo.upper()))[0])
        else:
            lines.append(plt.plot(data['new'][algo]['time_total_s'], data['new'][algo]['universal_reward_mean'], label=('new '+algo.upper()))[0])
            
    # Plot miscellaneous
    if optimum:
        plt.axhspan(optimum, optimum + 0.004, alpha=0.5, zorder=1, color='gold')
        lines.insert(0, Line2D([0], [0], label='[Nash equilibrium]', color='gold'))

    lower_anchor = 0.0 if game_type.name() == pu.games.MontyHall.name() else 0.07
    plt.legend(frameon=False, handles=lines, loc='lower right', ncol=3, bbox_to_anchor=(1.0, lower_anchor))

    # Plot configurations
    plt.xlim(0, 60)
    plt.ylim(0.0 if rcar else bottom, 0.5 if rcar else None)
        
    plt.title(f"{interaction_type.capitalize()} {game_type.name()} with {loss} loss")
    plt.xlabel("Total time in seconds")
    plt.ylabel("Universal reward mean")

    if rcar and eval:
        figure_path = f'figures/rcar_new_vs_old_eval/{game_type.name()}_{loss}_{interaction_type}.png'
    elif rcar:
        figure_path = f'figures/rcar_new_vs_old/{game_type.name()}_{loss}_{interaction_type}.png'
    elif old and new and eval:
        figure_path = f'figures/new_vs_old_eval/{game_type.name()}_{loss}_{interaction_type}.png'
    elif old and new:
        figure_path = f'figures/new_vs_old/{game_type.name()}_{loss}_{interaction_type}.png'
    elif new and eval:
        figure_path = f'figures/new_eval/{game_type.name()}_{loss}_{interaction_type}.png'
    elif new:
        figure_path = f'figures/new/{game_type.name()}_{loss}_{interaction_type}.png'
    else:
        figure_path = f'figures/old/{game_type.name()}_{loss}_{interaction_type}.png'

    plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.02)


def show_all(new: bool = True, old: bool = False, eval: bool = True, rcar: bool = False, algos: List[str] = marl.ALGOS):
    # Set style
    sns.set()
    
    if rcar:
        # Plot rcar figures
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type='zero-sum', rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type='zero-sum', rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type='zero-sum', rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type='zero-sum', rcar=rcar)

    else:
        # Plot mean reward figures
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', optimum=-0.66667)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='cooperative', optimum=-0.66667)
    
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type='zero-sum', optimum=-0.888)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type='cooperative', optimum=-0.666)
    
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type='zero-sum', optimum=-1.273, bottom=-2.5)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type='cooperative', optimum=-0.924, bottom=-2.5)
    
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='zero-sum', optimum=-1.333)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type='cooperative', optimum=-1.333)
    
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type='zero-sum', optimum=-1.444)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type='cooperative', optimum=-1.333)
    
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type='zero-sum', optimum=-2.659)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type='cooperative', optimum=-2.197)

    # Present plot(s)
    plt.show()


if __name__ == '__main__':
    show_all(
        new=True,
        old=True,
        eval=True,
        rcar=True,
        algos=[pu.PPO],
    )
