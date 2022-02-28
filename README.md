# Multi-agent reinforcement learning for relaxed probability updating games

This is the repository corresponding to the Master Thesis "Investigating relaxed probability updating games". It contains the code for the mechanics behind general-sum probability updating games and provides a wrapper to port such games to OpenAI Gym-like environments. Using Ray Tune and Ray RLlib, it implements PPO and other policy gradient learning methods in a decentralised manner on these games.

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

To run the main learning
