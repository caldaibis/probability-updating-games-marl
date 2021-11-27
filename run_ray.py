from __future__ import annotations

import inspect
import os
import json
from enum import Enum, auto
from typing import Type, Dict, Optional, List

from ray.rllib.agents import Trainer
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.tune.checkpoint_manager import Checkpoint
from ray.util.ml_utils.dict import merge_dicts

import probability_updating as pu
import probability_updating.games as games

import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune import Trainable, Callback
from ray.tune.trial import Trial

import supersuit as ss

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.ddpg import TD3Trainer, DDPGTrainer
from ray.rllib.agents.pg import PGTrainer


class RayModel(Enum):
    PPO = auto(),
    A2C = auto(),
    SAC = auto(),
    TD3 = auto(),
    DDPG = auto(),
    PG = auto(),


def get_trainable(model_type: RayModel) -> Type[Trainable]:
    return {
        RayModel.PPO: PPOTrainer,
        RayModel.A2C: A2CTrainer,
        RayModel.SAC: SACTrainer,
        RayModel.TD3: TD3Trainer,
        RayModel.DDPG: DDPGTrainer,
        RayModel.PG: PGTrainer,
    }[model_type]


def shared_parameter_env(game: games.Game) -> ParallelPettingZooEnv:
    env = pu.ProbabilityUpdatingEnv(game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)

    return ParallelPettingZooEnv(env)


def shared_critic_env(game: games.Game) -> ParallelPettingZooEnv:
    env = pu.ProbabilityUpdatingEnv(game)

    return ParallelPettingZooEnv(env)


def basic_config() -> dict:
    return {
        "env": "pug",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 10,
        "framework": "tf",  # "tf"
        "multiagent": {
            "policies": {"default_policy"},
            "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
        },
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_config": {
            "explore": False
        },
        # "custom_eval_function": custom_eval_function,
    }


def custom_eval_function(trainer: Trainer, eval_workers: WorkerSet):
    pass


def get_full_config(model: RayModel) -> dict:
    return {
        RayModel.PPO: merge_dicts(basic_config(), ppo_config()),
        RayModel.A2C: merge_dicts(basic_config(), ppo_config()),
        RayModel.SAC: merge_dicts(basic_config(), ppo_config()),
        RayModel.TD3: merge_dicts(basic_config(), ppo_config()),
        RayModel.DDPG: merge_dicts(basic_config(), ppo_config()),
        RayModel.PG: merge_dicts(basic_config(), ppo_config()),
    }[model]


def ppo_config() -> dict:
    # Todo: hyperparameter tuning!
    return {
        # "model": {
        #     "vf_share_layers": False,
        # },
        # "vf_loss_coeff": 0.01,
        # "train_batch_size": 10,
        # "sgd_minibatch_size": 1,
        # "num_sgd_iter": 30,
    }


class MyCallback(Callback):
    def on_step_begin(self, iteration: int, trials: List["Trial"], **info):
        print(inspect.stack()[0][3])

    def on_step_end(self, iteration: int, trials: List["Trial"], **info):
        print(inspect.stack()[0][3])

    def on_trial_start(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(inspect.stack()[0][3])

    def on_trial_restore(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(inspect.stack()[0][3])

    def on_trial_save(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(inspect.stack()[0][3])

    def on_trial_result(self, iteration: int, trials: List["Trial"], trial: "Trial", result: Dict, **info):
        print(inspect.stack()[0][3])
        print("Iteration", iteration)
        print("episode_reward_max", result["episode_reward_max"])
        print("episode_reward_min", result["episode_reward_min"])
        print("episode_reward_mean", result["episode_reward_mean"])
        print("stats", json.dumps(result["info"]["learner"], indent=2, default=str))
        print("evaluation", json.dumps(result["evaluation"], indent=2, default=str))

    def on_trial_complete(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(inspect.stack()[0][3])

    def on_trial_error(self, iteration: int, trials: List["Trial"], trial: "Trial", **info):
        print(inspect.stack()[0][3])

    def on_checkpoint(self, iteration: int, trials: List["Trial"], trial: "Trial", checkpoint: Checkpoint, **info):
        print(inspect.stack()[0][3])

    def on_experiment_end(self, trials: List["Trial"], **info):
        print(inspect.stack()[0][3])


def learn(game: games.Game, losses: Dict[pu.Agent, pu.Loss], model_type: RayModel, iterations: int, ext_name: Optional[str] = '') -> str:
    analysis = ray.tune.run(
        get_trainable(model_type),
        name=f"g={game.name()}_c={losses[pu.cont()].name}_q={losses[pu.quiz()].name}_iter={iterations}_e={ext_name}",
        config=get_full_config(model_type),
        stop={"training_iteration": iterations},  # "timesteps_total": 1, "episodes_total": 1
        checkpoint_freq=0,
        checkpoint_at_end=True,
        verbose=1,
        local_dir='./output_ray',
        callbacks=[MyCallback()],
    )

    return analysis.get_last_checkpoint()


def predict(game: games.Game, env: ParallelPettingZooEnv, model_type: RayModel, checkpoint: str):
    model = get_trainable(model_type)(config=get_full_config(model_type))
    model.restore(checkpoint)

    obs = env.reset()
    actions = {agent: model.compute_single_action(obs[agent], unsquash_action=True, explore=False) for agent in pu.agents()}

    obs, rewards, dones, infos = env.step(actions)

    return game
