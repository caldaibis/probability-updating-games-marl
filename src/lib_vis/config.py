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
    'independent_softmax': {
        'algos': [marl.PPO, marl.A2C, marl.TD3, marl.SAC],
        'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
        'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
        'interactions': [pu.COOPERATIVE, pu.ZERO_SUM],
    },
    'independent_punish_with_rcar_dist': {
        'algos': [marl.PPO, marl.A2C],
        'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
        'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
        'interactions': [pu.ZERO_SUM],
    },
    'independent_punish': {
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
    'legend-lower-anchor': 0.0,
    'title': True,
}

GRAPH_DIRICHLET = {
    **GRAPH_BASE,
    'name': 'dirichlet_new',
    'experiments': ['dirichlet'],
    **RUN_INFO['dirichlet'],
    'title': False,
    'legend-lower-anchor': 0.75,
    'show_figures': True,
    'save_figures': True,
    
    # 'legend-lower-anchor': {
    #     (pu_games.MONTY_HALL, pu.RANDOMISED_ZERO_ONE, pu.ZERO_SUM): 0.65,
    #     (pu_games.MONTY_HALL, pu.RANDOMISED_ZERO_ONE, pu.COOPERATIVE): 0.65,
    #
    #     (pu_games.MONTY_HALL, pu.BRIER, pu.ZERO_SUM): 0.65,
    #     (pu_games.MONTY_HALL, pu.BRIER, pu.COOPERATIVE): 0.65,
    #
    #     (pu_games.MONTY_HALL, pu.LOGARITHMIC, pu.ZERO_SUM): 0.30,
    #     (pu_games.MONTY_HALL, pu.LOGARITHMIC, pu.COOPERATIVE): 0.30,
    #
    #     (pu_games.FAIR_DIE, pu.RANDOMISED_ZERO_ONE, pu.ZERO_SUM): 0.50,
    #     (pu_games.FAIR_DIE, pu.RANDOMISED_ZERO_ONE, pu.COOPERATIVE): 0.62,
    #
    #     (pu_games.FAIR_DIE, pu.BRIER, pu.ZERO_SUM): 0.58,
    #     (pu_games.FAIR_DIE, pu.BRIER, pu.COOPERATIVE): 0.65,
    #
    #     (pu_games.FAIR_DIE, pu.LOGARITHMIC, pu.ZERO_SUM): 0.68,
    #     (pu_games.FAIR_DIE, pu.LOGARITHMIC, pu.COOPERATIVE): 0.70,
    # }
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

GRAPH_DIRICHLET_VS_INDEPENDENT_SOFTMAX_INDEPENDENT_PUNISH = {
    **GRAPH_BASE,
    'name': 'dirichlet_vs_independent_softmax_vs_independent_punish',
    'experiments': ['dirichlet', 'independent_softmax', 'independent_punish'],
    'algos': [marl.PPO],
    'games': [pu_games.MONTY_HALL, pu_games.FAIR_DIE],
    'losses': [pu.RANDOMISED_ZERO_ONE, pu.BRIER, pu.LOGARITHMIC],
    'interactions': [pu.COOPERATIVE, pu.ZERO_SUM],
    'title': False,
    'legend-lower-anchor': 0.83,
    # 'legend-lower-anchor': {
    #     (pu_games.MONTY_HALL, pu.RANDOMISED_ZERO_ONE, pu.ZERO_SUM): 0.65,
    #     (pu_games.MONTY_HALL, pu.RANDOMISED_ZERO_ONE, pu.COOPERATIVE): 0.65,
    #
    #     (pu_games.MONTY_HALL, pu.BRIER, pu.ZERO_SUM): 0.65,
    #     (pu_games.MONTY_HALL, pu.BRIER, pu.COOPERATIVE): 0.65,
    #
    #     (pu_games.MONTY_HALL, pu.LOGARITHMIC, pu.ZERO_SUM): 0.30,
    #     (pu_games.MONTY_HALL, pu.LOGARITHMIC, pu.COOPERATIVE): 0.30,
    #
    #     (pu_games.FAIR_DIE, pu.RANDOMISED_ZERO_ONE, pu.ZERO_SUM): 0.50,
    #     (pu_games.FAIR_DIE, pu.RANDOMISED_ZERO_ONE, pu.COOPERATIVE): 0.62,
    #
    #     (pu_games.FAIR_DIE, pu.BRIER, pu.ZERO_SUM): 0.58,
    #     (pu_games.FAIR_DIE, pu.BRIER, pu.COOPERATIVE): 0.65,
    #
    #     (pu_games.FAIR_DIE, pu.LOGARITHMIC, pu.ZERO_SUM): 0.68,
    #     (pu_games.FAIR_DIE, pu.LOGARITHMIC, pu.COOPERATIVE): 0.70,
    # },
    'show_figures': True,
    'save_figures': True,
}

GRAPH_INDEPENDENT_PUNISH = {
    **GRAPH_BASE,
    'metric': marl.REWARD_CONT,
    'name': 'independent_punish',
    'experiments': ['gaussian_box'],
    **RUN_INFO['independent_punish'],
    'legend-lower-anchor': 0.77,
    'title': False,
}

GRAPH_INDEPENDENT_SOFTMAX = {
    **GRAPH_BASE,
    'metric': marl.REWARD_CONT,
    'name': 'independent_softmax',
    'experiments': ['gaussian_softmax'],
    **RUN_INFO['independent_softmax'],
    'title': False,
    'legend-lower-anchor': {
        (pu_games.MONTY_HALL, pu.RANDOMISED_ZERO_ONE, pu.ZERO_SUM): 0.65,
        (pu_games.MONTY_HALL, pu.RANDOMISED_ZERO_ONE, pu.COOPERATIVE): 0.65,
        
        (pu_games.MONTY_HALL, pu.BRIER, pu.ZERO_SUM): 0.65,
        (pu_games.MONTY_HALL, pu.BRIER, pu.COOPERATIVE): 0.65,
        
        (pu_games.MONTY_HALL, pu.LOGARITHMIC, pu.ZERO_SUM): 0.30,
        (pu_games.MONTY_HALL, pu.LOGARITHMIC, pu.COOPERATIVE): 0.30,
        
        (pu_games.FAIR_DIE, pu.RANDOMISED_ZERO_ONE, pu.ZERO_SUM): 0.50,
        (pu_games.FAIR_DIE, pu.RANDOMISED_ZERO_ONE, pu.COOPERATIVE): 0.62,
        
        (pu_games.FAIR_DIE, pu.BRIER, pu.ZERO_SUM): 0.58,
        (pu_games.FAIR_DIE, pu.BRIER, pu.COOPERATIVE): 0.65,
        
        (pu_games.FAIR_DIE, pu.LOGARITHMIC, pu.ZERO_SUM): 0.68,
        (pu_games.FAIR_DIE, pu.LOGARITHMIC, pu.COOPERATIVE): 0.70,
    }
}
