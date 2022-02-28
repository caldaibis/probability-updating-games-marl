# Multi-agent reinforcement learning for relaxed probability updating games

Welcome to the repository corresponding to the master thesis project "Investigating relaxed probability updating games" by Collin Aldaibis. This repository is built and utilised to investigate Nash equilibria for general-sum probability updating games. It implements all of the mechanics in such games and provides a wrapper to port them to OpenAI Gym-like environments. Using Ray Tune and Ray RLlib, it applies the state-of-the-art Proximal Policy Optimisation (PPO) and other policy gradient learning methods to learn optimal strategies for both the host and the contestant. This implementation empirically shows to converge to Nash equilibria for many general-sum games.

## Requirements and notes

This software has exclusively been run and tested with
- Windows 11
- Python 3.8 64-bits
We therefore recommend using these when executing this code. It is possible that other settings may not work.

## Install

Suppose you have cloned this repository and cd'ed into the repository folder. We recommend you to setup a python virtual environment:
```
python3 -m venv env
```

Subsequently, you will want to activate your virtual environment. On Windows, you can do this as follows ():
```
env\Scripts\activate
```

Now, install the list of required packages:
```
pip install -r requirements.txt
```

This should install all the dependencies within the virtual environment to run the code in this repository.

## Run

The main module is src.main.main. From here, any general-sum game can be trained with any of the implemented algorithms. Note that this file is run with a certain set of parameters, in a specific order. The ordered list of parameters, including description and example values, is as follows: 
```
{
  'algorithm': 'ppo', # or 'a2c', 'ddpg', 'td3' and 'sac'
  'game': 'monty_hall', # or 'fair_die' and 'example_c' through 'example_h'
  'cont': 'logarithmic', # or 'brier', 'randomised_0_1', 'matrix_predefined_pos_0' through 'matrix_predefined_pos_14', 'matrix_predefined_neg_0' through 'matrix_predefined_neg_14'
  'host': 'logarithmic_neg', # you can also use 'brier_neg', 'randomised_0_1_neg' and all other loss functions
  'show_example': 1, # Shows an example with default strategies
  'debug_mode': 0, # Enables debug mode
  'ray': 0, # Enables the Ray framework to learn or load
  'learn': 1, # If 'ray' = 1, learns the game
  'show_figure': 1, # If 'ray' = 1, displays matplotlib graphs of learned or loaded run
  'show_eval': 1, # If 'ray' = 1, shows graph evaluation metrics instead of default metrics
  'save_figures': 1, # If 'ray' = 1, saves matplotlib graphs of learned or loaded run to file system
  'save_progress': 1, # If 'ray' = 1, saves training progress and performance to file system
  'max_total_time_s': 80 # Defines the runtime of the current run
}
```
With the above settings, we get the following command:
```
py -m src.main.main ppo monty_hall logarithmic logarithmic_neg 1 0 0 1 0 1 1 0 80
```
Accordingly, a PPO model will be trained for exactly 80 seconds on **Monty Hall** with logarithmic loss for the contestant and negative logarithmic loss for the host. Subsequently, the resulting training progression data, performance and strategy graphs will automatically be saved to the file system, for potential investigative purposes.
