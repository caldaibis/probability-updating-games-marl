from __future__ import annotations

import inspect
from typing import List, Dict

from ray.tune import Callback
from ray.tune.checkpoint_manager import Checkpoint


class CustomCallback(Callback):
    def on_step_begin(self, iteration: int, trials: List["Trial"], **info):
        pass  # print(inspect.stack()[0][3])

    def on_step_end(self, iteration: int, trials: List["Trial"], **info):
        pass  # print(inspect.stack()[0][3])

    def on_trial_start(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        pass  # print(inspect.stack()[0][3])

    def on_trial_restore(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        pass  # print(inspect.stack()[0][3])

    def on_trial_save(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        pass  # print(inspect.stack()[0][3])

    def on_trial_result(self, iteration: int, trials: List["Trial"], trial: "Trial", result: Dict, **info):
        pass  # print(inspect.stack()[0][3])
        # print("episode_reward_max", result["episode_reward_max"])
        # print("episode_reward_min", result["episode_reward_min"])
        # print("stats", json.dumps(result["info"]["learner"], indent=2, default=str))

        # print("episode_reward_mean", result["episode_reward_mean"])
        # print("evaluation", result["evaluation"]["episode_reward_mean"])

    def on_trial_complete(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        pass  # print(inspect.stack()[0][3])

    def on_trial_error(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        pass  # print(inspect.stack()[0][3])

    def on_checkpoint(self, iteration: int, trials: List["Trial"], trial: "Trial", checkpoint: Checkpoint, **info):
        pass  # print(inspect.stack()[0][3])

    def on_experiment_end(self, trials: List["Trial"], **info):
        pass  # print(inspect.stack()[0][3])
