# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

from __future__ import annotations

import os
from typing import Type, Dict

import probability_updating as pu
import probability_updating.games as games

import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import ray.rllib.agents.ppo as ppo
from ray.tune import register_env, Trainable
from ray.tune.logger import pretty_print

import supersuit as ss


env_name = "probability_updating"


def create_ray_env(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss]):
    env = pu.probability_updating_env.env(game_type(losses))
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)

    return ParallelPettingZooEnv(env)


def plain_config() -> dict:
    return {
        "env": env_name,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 7,
        "framework": "torch",  # "tf"
        "log_level": "DEBUG",
        "multiagent": {
            "policies": {"default_policy"},
            "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
        },
    }


def ppo_config() -> dict:
    # Todo: hyperparameter tuning!
    return {
        **plain_config(),
        "model": {
            "vf_share_layers": True,
        },
        "vf_loss_coeff": 0.001,
        # "train_batch_size": 4000,
        # "sgd_minibatch_size": 20,
        # "num_sgd_iter": 30,
    }


def setup(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainable]) -> Trainable:
    env = create_ray_env(game_type, losses)
    register_env(env_name, lambda _: env)

    return trainer_type(config=ppo_config())


def learn(trainer: Trainable) -> str:
    result = None
    for i in range(25):
        # Perform one iteration of training the policy with the network
        result = trainer.train()
        print("Training iteration", i)

    checkpoint = trainer.save('./ray_dir/without_tune')
    print(pretty_print(result))

    return checkpoint


def predict(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], trainer: Trainable) -> (dict, dict):
    env = create_ray_env(game_type, losses)
    obs = env.reset()

    actions = {agent: trainer.compute_single_action(obs[agent], unsquash_action=True, explore=False) for agent in pu.agents()}

    # output results
    print("Actions")
    print(f"action cont {actions[pu.cont()]}")
    print(f"action quiz {actions[pu.quiz()]}")
    print()

    obs, rewards, dones, infos = env.step(actions)

    return actions, rewards


def learn_with_tune(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainable]):
    env = create_ray_env(game_type, losses)
    register_env(env_name, lambda _: env)

    analysis = ray.tune.run(
        trainer_type,
        config=ppo_config(),
        stop={"training_iteration": 10},  # "episodes_total": 100
        checkpoint_freq=0,
        checkpoint_at_end=True,
        verbose=3,
        local_dir='./ray_dir',
    )

    # best_checkpoint = analysis.get_last_checkpoint(metric="episode_reward_mean", mode="max")
    return analysis.get_last_checkpoint()


def run(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss]):
    # Zet local_mode=True om te debuggen
    ray.init(local_mode=False)

    # Use Ray Tune to learn model
    checkpoint = learn_with_tune(game_type, losses, ppo.PPOTrainer)

    # # Learn directly through API
    # agent = setup(game_type, losses, ppo.PPOTrainer)
    # checkpoint = learn(trainer)
    # # OR just load model from disk
    # checkpoint = "./ray_dir/without_tune\checkpoint_000025\checkpoint-25"

    # Using the model, predict the optimal strategy
    agent = ppo.PPOTrainer(config=ppo_config())
    agent.restore(checkpoint)
    actions, rewards = predict(game_type, losses, agent)

    print("Rewards")
    print(f"reward cont {rewards[pu.cont()]}")
    print(f"reward quiz {rewards[pu.quiz()]}")
    print()
