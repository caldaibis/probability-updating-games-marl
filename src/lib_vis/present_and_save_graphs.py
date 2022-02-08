from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Type, List, Dict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

import src.lib_marl as marl
import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_vis as vis


def plot(algos: List[str], new: bool, old: bool, eval: bool, game_type: Type[pu_games.Game], loss: str, interaction_type: str, optimum: Optional[float] = None, bottom: Optional[float] = None, rcar: bool = False):
    plt.figure()

    # Load datasets
    data = {'old': {}, 'new': {}}
    for algo in algos:
        try:
            if new:
                data['new'][algo] = pd.read_csv(
                    f'data/dirichlet_{game_type.name()}_{loss}_{interaction_type}_{algo}.csv')
            if (old and rcar) or (new and old):
                data['old'][algo] = pd.read_csv(f'data/old_rcar_dist/{game_type.name()}/{loss}/{interaction_type}/{algo}.csv')
            elif old:
                data['old'][algo] = pd.read_csv(
                    f'data/gaussian_box/{loss}/{game_type.name()}/{interaction_type}/{algo}.csv')
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
        figure_path = f'figures/dirichlet/{game_type.name()}_{loss}_{interaction_type}.png'
    else:
        figure_path = f'figures/gaussian_box/{game_type.name()}_{loss}_{interaction_type}.png'

    plt.savefig(figure_path, transparent=False, bbox_inches='tight', pad_inches=0.02)


def show_all(new: bool = True, old: bool = False, eval: bool = True, rcar: bool = False, algos: List[str] = marl.ALGOS):
    if rcar:
        # Plot rcar figures
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type=pu.ZERO_SUM, rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type=pu.ZERO_SUM, rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type=pu.ZERO_SUM, rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type=pu.ZERO_SUM, rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type=pu.ZERO_SUM, rcar=rcar)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type=pu.ZERO_SUM, rcar=rcar)

    else:
        # Plot mean reward figures
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type=pu.ZERO_SUM, optimum=-0.66667)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.RANDOMISED_ZERO_ONE, interaction_type=pu.COOPERATIVE, optimum=-0.66667)
    
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type=pu.ZERO_SUM, optimum=-0.888)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.BRIER, interaction_type=pu.COOPERATIVE, optimum=-0.666)
    
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type=pu.ZERO_SUM, optimum=-1.273, bottom=-2.5)
        plot(algos, new, old, eval, game_type=pu.games.MontyHall, loss=pu.LOGARITHMIC, interaction_type=pu.COOPERATIVE, optimum=-0.924, bottom=-2.5)
    
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type=pu.ZERO_SUM, optimum=-1.333)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.RANDOMISED_ZERO_ONE, interaction_type=pu.COOPERATIVE, optimum=-1.333)
    
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type=pu.ZERO_SUM, optimum=-1.444)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.BRIER, interaction_type=pu.COOPERATIVE, optimum=-1.333)
    
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type=pu.ZERO_SUM, optimum=-2.659)
        plot(algos, new, old, eval, game_type=pu.games.FairDie, loss=pu.LOGARITHMIC, interaction_type=pu.COOPERATIVE, optimum=-2.197)

    # Present plot(s)
    plt.show()


def run(config: Dict):
    root_dir = Path(f'figures/{config["name"]}/')
    if config['save_figures']:
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        
    for game in config['games']:
        for loss in config['losses']:
            for interaction in config['interactions']:
                # initiate new figure
                plt.figure()
                
                # load dataframes and draw lines
                lines = []
                x_max = 0
                for experiment in config['experiments']:
                    for algo in config['algos']:
                        df = pd.read_csv(f'data/{experiment}/{game}_{loss}_{interaction}_{algo}.csv')
                        
                        label = ''
                        if len(config['experiments']) > 1:
                            label = experiment
                        if len(config['algos']) > 1:
                            label = algo.upper()
                        
                        metric = config['metric']
                        if metric not in df.columns:
                            metric = marl.OLD_METRICS[metric]
                            
                        if metric in marl.NEGATIVE_METRICS:
                            df[metric] = -1*df[metric]
                        
                        lines.append(plt.plot(df['time_total_s'], df[metric], label=label)[0])
                        
                        # update max coordinate of the x axis
                        x_max = max(x_max, df.iloc[-1]['time_total_s'])
                        
                # plot miscellaneous
                if config['metric'] not in [marl.RCAR_DIST, marl.RCAR_DIST_EVAL] and (game, loss, interaction) in vis.NASH_EQUILIBRIA:
                    n = vis.NASH_EQUILIBRIA[(game, loss, interaction)]
                    lines.insert(0, plt.plot([0, 1000], [n, n], label='[NE]', color="black")[0])
                
                # build legend
                ncol = max(round(max(len(config['experiments']), len(config['algos'])) / 2.0 + 0.0), 1)
                if isinstance(config['legend-lower-anchor'], dict):
                    plt.legend(frameon=False, handles=lines, loc='lower right', ncol=ncol, bbox_to_anchor=(1.0, config['legend-lower-anchor'][(game, loss, interaction)]))
                else:
                    plt.legend(frameon=False, handles=lines, loc='lower right', ncol=ncol, bbox_to_anchor=(1.0, config['legend-lower-anchor']))

                # plot config
                plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
                plt.xlim(0, x_max)
                # plt.ylim(top=0.0)
                # plt.ylim(0.0)
                
                if config['title']:
                    plt.title(f"{pu.INTERACTIONS[interaction].capitalize()} {pu_games.GAME_PRETTY_NAMES[game]} with {pu.LOSS_NAMES[loss]} loss")
                plt.xlabel("Total time in seconds")
                plt.ylabel(marl.ALL_METRICS[config['metric']])

                # save figure to filesystem
                if config['save_figures']:
                    plt.savefig(f'figures/{config["name"]}/{game}_{loss}_{interaction}.png', transparent=False, bbox_inches='tight', pad_inches=0.02)
     
    # present all figures
    if config['show_figures']:
        plt.show()
        
        
if __name__ == '__main__':
    sns.set_theme(color_codes=True)
    
    configuration = {
        **vis.GRAPH_DIRICHLET_VS_INDEPENDENT_SOFTMAX_INDEPENDENT_PUNISH,
    }
    
    run(configuration)
    
    # show_all(
    #     new=True,
    #     old=True,
    #     eval=True,
    #     rcar=True,
    #     algos=[marl.PPO],
    # )
