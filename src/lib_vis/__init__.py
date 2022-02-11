from __future__ import annotations

from typing import Tuple

import numpy as np

from src.lib_vis.config import *
from src.lib_vis.present_and_save_graphs import *
from src.lib_vis.per_trial import *
from src.lib_vis.trial_aggregation import *
from src.lib_vis.strategy_scatterplots import *

import src.lib_pu.games as pu_games

GAME_LOSS_Y_STEP = {
    pu_games.MONTY_HALL: 1/6,
    pu_games.FAIR_DIE: 1/6,
    pu_games.EXAMPLE_C: 1/6,
    pu_games.EXAMPLE_D: 1/6,
    pu_games.EXAMPLE_E: 1/6,
    pu_games.EXAMPLE_F: 1/6,
    pu_games.SQUARE: 1/6,
    pu_games.EXAMPLE_H: 1/6,
}

GAME_RCAR_Y_MAX = {
    pu_games.MONTY_HALL: 0.36,
    pu_games.FAIR_DIE: 0.36,
    pu_games.EXAMPLE_C: 0.36,
    pu_games.EXAMPLE_D: 0.36,
    pu_games.EXAMPLE_E: 0.36,
    pu_games.EXAMPLE_F: 0.36,
    pu_games.SQUARE: 0.36,
    pu_games.EXAMPLE_H: 0.36,
}


def get_y_min_max(df_metric_list: List[Tuple[str, List[pd.DataFrame]]], config: Dict) -> Tuple:
    y_min = min(df[metric].min() for (metric, dfs) in df_metric_list for df in dfs)
    y_max = max(df[metric].max() for (metric, dfs) in df_metric_list for df in dfs)
    
    diff = y_max - y_min
    y_min -= diff * 0.05
    y_max += diff * 0.05
    
    return y_min, max(y_max, vis.GAME_RCAR_Y_MAX[config['game']])


def get_y_ticks(config: Dict, metrics: List[str]) -> Optional[np.ndarray]:
    if any(m in marl.NEGATIVE_METRICS for m in metrics):
        return np.arange(-6, 6, vis.GAME_LOSS_Y_STEP[config['game']])
    return None


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


def init():
    sns.set_theme(color_codes=True)
