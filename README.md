# Multi-agent reinforcement learning for relaxed probability updating games

This is the repository corresponding to the Master Thesis "Investigating relaxed probability updating games" by Collin Aldaibis. It contains the code for the mechanics behind general-sum probability updating games and provides a wrapper to port such games to OpenAI Gym-like environments. Using Ray Tune and Ray RLlib, it implements PPO and other policy gradient learning methods to learn optimal strategies, converging to Nash equilibria.

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

The main file is src/main/main.py. This file is run with the certain parameters set in a specific order. The ordered list of parameters, including description and example values is given below:
```
{
  'algorithm': 'ppo', # or 'a2c', 'ddpg', 'td3' and 'sac'
  'game': 'monty_hall', # or 'fair_die' and 'example_c' through 'example_h'
  'cont': 'logarithmic', # or 'brier', 'randomised_0_1', 'matrix_predefined_pos_0' through 'matrix_predefined_pos_14', 'matrix_predefined_neg_0' through 'matrix_predefined_neg_14'
  'host': 'logarithmic_neg', # you can also use 'brier_neg', 'randomised_0_1_neg' and all loss functions
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
Accordingly, a PPO model will be trained for exactly 80 seconds on **Monty Hall** with $L_C$ logarithmic and $L_H$ negative logarithmic. Subsequently, the resulting training progression data, performance and strategy graphs will automatically be saved to the file system for investigation.
