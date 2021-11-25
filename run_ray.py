from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Type, Dict, Optional

from ray.rllib.policy.policy import PolicySpec
from ray.util.ml_utils.dict import merge_dicts

import probability_updating as pu
import games

import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune import register_env

import supersuit as ss

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.ddpg import TD3Trainer, DDPGTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.contrib.maddpg import MADDPGTrainer


env_name = "pug"


class RayModel(Enum):
    PPO = PPOTrainer,
    A2C = A2CTrainer,
    SAC = SACTrainer,
    TD3 = TD3Trainer,
    DDPG = DDPGTrainer,
    PG = PGTrainer,
    MADDPG = MADDPGTrainer


def shared_parameter_env(game: games.Game) -> ParallelPettingZooEnv:
    env = pu.probability_updating_env.env(game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)

    return ParallelPettingZooEnv(env)


def shared_critic_env(game: games.Game) -> ParallelPettingZooEnv:
    env = pu.probability_updating_env.env(game)

    return ParallelPettingZooEnv(env)


def basic_config() -> dict:
    return {
        "env": env_name,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 9,
        "framework": "torch",  # "tf"
    }


def parameter_sharing_config() -> dict:
    return {
        "multiagent": {
            "policies": {"default_policy"},
            "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
        },
    }


def shared_critic_config(env: ParallelPettingZooEnv) -> dict:
    return {
        # "multiagent": {
        #     "policies": {"default_policy"},
        #     "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
        # },
        "multiagent": {
            "policies": {
                pu.cont(): PolicySpec(
                    observation_space=env.observation_spaces[pu.cont()],
                    action_space=env.action_spaces[pu.cont()],
                    config={"agent_id": 0}),
                pu.quiz(): PolicySpec(
                    observation_space=env.observation_spaces[pu.quiz()],
                    action_space=env.action_spaces[pu.quiz()],
                    config={"agent_id": 1}),
            },
            "policy_mapping_fn": (
                lambda agent_id, **kwargs: {0: pu.cont(), 1: pu.quiz()}[agent_id]),
        }
    }


def get_full_config(model: RayModel, env: ParallelPettingZooEnv) -> dict:
    return {
        RayModel.PPO: merge_dicts(basic_config(), merge_dicts(parameter_sharing_config(), ppo_config())),
        RayModel.A2C: merge_dicts(basic_config(), merge_dicts(parameter_sharing_config(), ppo_config())),
        RayModel.SAC: merge_dicts(basic_config(), merge_dicts(parameter_sharing_config(), ppo_config())),
        RayModel.TD3: merge_dicts(basic_config(), merge_dicts(parameter_sharing_config(), ppo_config())),
        RayModel.DDPG: merge_dicts(basic_config(), merge_dicts(parameter_sharing_config(), ppo_config())),
        RayModel.PG: merge_dicts(basic_config(), merge_dicts(parameter_sharing_config(), ppo_config())),
        RayModel.MADDPG: merge_dicts(basic_config(), merge_dicts(shared_critic_config(env), ppo_config())),
    }[model]


def ppo_config() -> dict:
    # Todo: hyperparameter tuning!
    return {
        # "model": {
        #     "vf_share_layers": False,
        # },
        # "vf_loss_coeff": 0.01,
        # "train_batch_size": 4000,
        # "sgd_minibatch_size": 20,
        # "num_sgd_iter": 30,
    }


def init(game: games.Game) -> ParallelPettingZooEnv:
    ray.shutdown()
    # Zet local_mode=True om te debuggen
    ray.init(local_mode=False, logging_level=logging.DEBUG)

    # Setup environment
    env = shared_parameter_env(game)
    register_env(env_name, lambda _: env)

    return env


def learn(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], model_type: RayModel, iterations: int, ext_name: Optional[str] = '') -> str:
    env = init(game_type(losses))

    analysis = ray.tune.run(
        model_type.value,
        name=f"g={game_type.name()}_c={losses[pu.cont()].name}_q={losses[pu.quiz()].name}_iter={iterations}_e={ext_name}",
        config=get_full_config(model_type, env),
        stop={"training_iteration": iterations},
        checkpoint_freq=0,
        checkpoint_at_end=True,
        verbose=3,
        local_dir='./output_ray',
    )

    # best_checkpoint = analysis.get_last_checkpoint(metric="episode_reward_mean", mode="max")
    return analysis.get_last_checkpoint()


def predict(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], model_type: RayModel, checkpoint: str) -> games.Game:
    game = game_type(losses)
    env = init(game)

    model = model_type.value(config=get_full_config(model_type, env))
    model.restore(checkpoint)

    obs = env.reset()
    actions = {agent: model.compute_single_action(obs[agent], unsquash_action=True, explore=False) for agent in pu.agents()}

    obs, rewards, dones, infos = env.step(actions)

    return game
