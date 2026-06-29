# Multi-agent reinforcement learning for relaxed probability updating games

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

Welcome to the repository corresponding to the master thesis project "Investigating relaxed probability updating games" by Collin Aldaibis. This repository is built and utilised to investigate Nash equilibria for general-sum probability updating games. It implements all of the mechanics in such games and provides a wrapper to port them to OpenAI Gym-like environments. Using Ray Tune and Ray RLlib, it applies the state-of-the-art Proximal Policy Optimisation (PPO) and other policy gradient learning methods to learn optimal strategies for both the host and the contestant. This implementation empirically shows to converge to Nash equilibria for many general-sum games.

## Repository layout

| Path | Description |
| --- | --- |
| `src/lib_pu` | Core probability updating game mechanics and the Gym-like environment wrappers. |
| `src/lib_pu/games` | Concrete game definitions (Monty Hall, fair die, examples C–H, ...). |
| `src/lib_marl` | Multi-agent RL glue: model wrapper, custom metrics, stoppers and Ray constants. |
| `src/lib_vis` | Plotting and result aggregation utilities. |
| `src/main` | Entry points (`main.py`), batch run scripts and saved matrices. |

## Requirements and notes

This software was developed and tested with
- Windows 11
- Python 3.8 (64-bit)

These are the recommended settings. The code is plain Python and should also run on Linux and macOS, though it has not been extensively tested there and some of the pinned 2022-era dependencies may need adjusting.

## Install

After cloning the repository and changing into its folder, set up a virtual environment:
```
python3 -m venv venv
```

Activate it:
```
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Install the required packages:
```
pip install -r requirements.txt
```

This installs all the dependencies needed to run the code in this repository.

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
With the above settings, we get the following command (run it from the repository root):
```
python -m src.main.main ppo monty_hall logarithmic logarithmic_neg 1 0 0 1 0 1 1 0 80
```
Accordingly, a PPO model will be trained for exactly 80 seconds on **Monty Hall** with logarithmic loss for the contestant and negative logarithmic loss for the host. Subsequently, the resulting training progression data, performance and strategy graphs will automatically be saved to the file system, for potential investigative purposes.

## Citation

If you use this code in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).
