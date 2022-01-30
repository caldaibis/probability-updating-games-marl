from __future__ import annotations

import src.lib_pu as pu
import src.lib_pu.games as pu_games
import src.lib_marl as marl


NASH_EQUILIBRIA = {
    (pu_games.MONTY_HALL, pu.RANDOMISED_ZERO_ONE, pu.ZERO_SUM): 0.333,
    (pu_games.MONTY_HALL, pu.RANDOMISED_ZERO_ONE, pu.COOPERATIVE): 0.333,
    
    (pu_games.MONTY_HALL, pu.BRIER, pu.ZERO_SUM): 0.444,
    (pu_games.MONTY_HALL, pu.BRIER, pu.COOPERATIVE): 0.333,
    
    (pu_games.MONTY_HALL, pu.LOGARITHMIC, pu.ZERO_SUM): 0.636,
    (pu_games.MONTY_HALL, pu.LOGARITHMIC, pu.COOPERATIVE): 0.462,
    
    (pu_games.FAIR_DIE, pu.RANDOMISED_ZERO_ONE, pu.ZERO_SUM): 0.666,
    (pu_games.FAIR_DIE, pu.RANDOMISED_ZERO_ONE, pu.COOPERATIVE): 0.666,
    
    (pu_games.FAIR_DIE, pu.BRIER, pu.ZERO_SUM): 0.722,
    (pu_games.FAIR_DIE, pu.BRIER, pu.COOPERATIVE): 0.666,
    
    (pu_games.FAIR_DIE, pu.LOGARITHMIC, pu.ZERO_SUM): 1.329,
    (pu_games.FAIR_DIE, pu.LOGARITHMIC, pu.COOPERATIVE): 1.098,
}

RUN_INFO = {
    'dirichlet': {
        'algos': [marl.PPO, marl.A2C],
        'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
        'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
        'interactions': [pu.COOPERATIVE, pu.ZERO_SUM],
    },
    'gaussian_softmax': {
        'algos': [marl.PPO, marl.A2C, marl.TD3, marl.SAC],
        'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
        'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
        'interactions': [pu.COOPERATIVE, pu.ZERO_SUM],
    },
    'gaussian_box_rcar_dist': {
        'algos': [marl.PPO, marl.A2C],
        'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
        'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
        'interactions': [pu.ZERO_SUM],
    },
    'gaussian_box': {
        'algos': [marl.PPO, marl.A2C, marl.DDPG, marl.TD3, marl.SAC],
        'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
        'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
        'interactions': [pu.COOPERATIVE, pu.ZERO_SUM],
    },
}

GRAPH_BASE = {
    'algos': [marl.PPO],
    'metric': marl.REWARD_CONT,
    'show_figures': True,
    'save_figures': False,
}

GRAPH_DIRICHLET = {
    **GRAPH_BASE,
    'name': 'dirichlet_2',
    'experiments': ['dirichlet'],
    'algos': [marl.PPO, marl.A2C],
    'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
    'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
    'interactions': [pu.COOPERATIVE, pu.ZERO_SUM],
}

GRAPH_DIRICHLET_VS_GAUSSIAN_SOFTMAX = {
    **GRAPH_BASE,
    'name': 'dirichlet_vs_gaussian_softmax',
    'experiments': ['dirichlet', 'gaussian_softmax'],
    'algos': [marl.PPO],
    'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
    'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
    'interactions': [pu.COOPERATIVE, pu.ZERO_SUM],
}

GRAPH_DIRICHLET_VS_GAUSSIAN_SOFTMAX_GAUSSIAN_BOX = {
    **GRAPH_BASE,
    'name': 'dirichlet_vs_gaussian_softmax_vs_gaussian_box',
    'experiments': ['dirichlet', 'gaussian_softmax', 'gaussian_box'],
    'algos': [marl.PPO],
    'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
    'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
    'interactions': [pu.COOPERATIVE, pu.ZERO_SUM],
}
