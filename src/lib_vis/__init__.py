from __future__ import annotations

from typing import Tuple

from src.lib_vis.config import *
from src.lib_vis.present_and_save_graphs import *
from src.lib_vis.per_trial import *
from src.lib_vis.trial_aggregation import *
from src.lib_vis.strategy_scatterplots import *

import src.lib_pu.games as pu_games


def get_y_step(game, metric) -> float:
    d = {
        pu_games.MONTY_HALL: {
            marl.REWARD_CONT: 1/9,
        },
        pu_games.FAIR_DIE: {
            marl.REWARD_CONT: 1/9,
        },
    }
    val = d.get(game)
    if val:
        val = val.get(metric)
    return val or 0.1


def get_y_lim(game, metric, loss=None) -> Optional[Tuple[float, float]]:
    d = {
        pu_games.MONTY_HALL: {
            marl.RCAR_DIST: {
                None: (-0.03, 0.36)
            },
        },
        pu_games.EXAMPLE_F: {
            marl.REWARD_CONT: {
                None: (-0.8, 0.8),
                pu.LOGARITHMIC: (0.6, 0.8),
                pu.LOGARITHMIC_NEG: (-0.8, -0.6),
            },
        },
    }
    val = d.get(game)
    if val:
        val = val.get(metric)
    if val:
        val = val.get(loss)
    return val


def get_main_metric(metrics: List[str]):
    if any(m in [marl.RCAR_DIST, marl.RCAR_DIST_EVAL] for m in metrics):
        return marl.RCAR_DIST
    if any(m in [marl.REWARD_CONT, marl.REWARD_CONT_EVAL, marl.REWARD_HOST, marl.REWARD_HOST_EVAL] for m in metrics):
        return marl.REWARD_CONT
    return metrics[0]


def get_y_min_max(df_metric_list: List[Tuple[str, List[pd.DataFrame]]], config: Dict, agent=None) -> Tuple:
    y_min = min(df[metric].min() for (metric, dfs) in df_metric_list for df in dfs)
    y_max = max(df[metric].max() for (metric, dfs) in df_metric_list for df in dfs)
    
    diff = y_max - y_min
    y_min -= diff * 0.05
    y_max += diff * 0.05
    
    main_metric = get_main_metric([m for (m, _) in df_metric_list])
    predefined_y_lim = get_y_lim(config['game'], main_metric, config.get(agent))
    
    if predefined_y_lim is not None:
        y_min = min(y_min, predefined_y_lim[0])
        y_max = max(y_max, predefined_y_lim[1])
        
    return y_min, y_max


def get_y_ticks(config: Dict, metrics: List[str]) -> Optional[np.ndarray]:
    predefined_y_step = get_y_step(config['game'], get_main_metric(metrics))
    if predefined_y_step is not None:
        return np.arange(-6, 6, predefined_y_step)
    return None


def set_subplot_figure(config: Dict, axs: List, figure_config: Dict, plot_configs: List[Dict]):
    for i in range(len(axs)):
        if 'legend' in plot_configs[i] and plot_configs[i]['legend']:
            axs[i].legend(facecolor='white')
        if 'title' in plot_configs[i]:
            axs[i].set_title(plot_configs[i]['title'])
        if 'xlabel' in plot_configs[i]:
            axs[i].set_xlabel(plot_configs[i]['xlabel'])
        if 'ylabel' in plot_configs[i]:
            axs[i].set_ylabel(plot_configs[i]['ylabel'])
        if 'xlim' in plot_configs[i]:
            axs[i].set_xlim(plot_configs[i]['xlim'])
        if 'yticks' in plot_configs[i] and plot_configs[i]['yticks'] is not None:
            axs[i].set_yticks(plot_configs[i]['yticks'])
        if 'ylim' in plot_configs[i] and plot_configs[i]['ylim'] is not None:
            axs[i].set_ylim(plot_configs[i]['ylim'])

    if config['save_figures']:
        d = f'figures/{config["game"]}/{config["t"]}_{config[pu.CONT]}_{config[pu.HOST]}/{figure_config["directory"]}'
        if not os.path.isdir(d):
            os.makedirs(d)
        plt.savefig(f'{d}/{figure_config["filename"]}', transparent=False, bbox_inches='tight', pad_inches=0.02)
    
    plt.figure()


def set_figure(config: dict, plot_config: dict):
    plt.title(plot_config['title'])
    plt.xlabel(plot_config['x_label'])
    plt.ylabel(plot_config['y_label'])
    if plot_config['legend']:
        plt.legend(facecolor='white')
    if plot_config['y_ticks'] is not None:
        plt.yticks(plot_config['y_ticks'])
    if plot_config['y_lim'] is not None:
        plt.ylim(*plot_config['y_lim'])

    if config['save_figures']:
        d = f'figures/{config["game"]}/{config["t"]}_{config[pu.CONT]}_{config[pu.HOST]}/{plot_config["directory"]}'
        if not os.path.isdir(d):
            os.makedirs(d)
        plt.savefig(f'{d}/{plot_config["filename"]}', transparent=False, bbox_inches='tight', pad_inches=0.02)
    
    plt.figure()


def init():
    sns.set_theme(color_codes=True)
    plt.figure()
