from __future__ import annotations

import logging
import os
from typing import Type, Dict, Optional

import probability_updating as pu
import probability_updating.games as games

import ray
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune import register_env, Trainable

import supersuit as ss


env_name = "pug"


def create_ray_env(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss]) -> ParallelPettingZooEnv:
    env = pu.probability_updating_env.env(game_type(losses))
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)

    return ParallelPettingZooEnv(env)


def plain_config() -> dict:
    return {
        "env": env_name,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 9,
        "framework": "torch",  # "tf"
        "multiagent": {
            "policies": {"default_policy"},
            "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
        },
    }


def ppo_config() -> dict:
    # Todo: hyperparameter tuning!
    return {
        **plain_config(),
        # "model": {
        #     "vf_share_layers": False,
        # },
        # "vf_loss_coeff": 0.01,
        # "train_batch_size": 4000,
        # "sgd_minibatch_size": 20,
        # "num_sgd_iter": 30,
    }


# def learn(trainer: Trainable) -> str:
#     result = None
#     for i in range(25):
#         # Perform one iteration of training the policy with the network
#         result = trainer.train()
#         print("Training iteration", i)
#
#     checkpoint = trainer.save('./ray_dir/without_tune')
#     print(pretty_print(result))
#
#     return checkpoint
#
#
# def predict(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], trainer: Trainable) -> (dict, dict):
#     env = create_ray_env(game_type, losses)
#     obs = env.reset()
#
#     actions = {agent: trainer.compute_single_action(obs[agent], unsquash_action=True, explore=False) for agent in pu.agents()}
#
#     # output results
#     print("Actions")
#     print(f"action cont {actions[pu.cont()]}")
#     print(f"action quiz {actions[pu.quiz()]}")
#     print()
#
#     obs, rewards, dones, infos = env.step(actions)
#
#     return actions, rewards

def init(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss]) -> ParallelPettingZooEnv:
    ray.shutdown()
    # Zet local_mode=True om te debuggen
    ray.init(local_mode=False, logging_level=logging.DEBUG)

    # Setup environment
    env = create_ray_env(game_type, losses)
    register_env(env_name, lambda _: env)

    return env


def learn(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainable], iterations: int, ext_name: Optional[str] = '') -> str:
    init(game_type, losses)

    analysis = ray.tune.run(
        trainer_type,
        name=f"g={game_type.name()}_c={losses[pu.cont()].name}_q={losses[pu.quiz()].name}_iter={iterations}_e={ext_name}",
        config=ppo_config(),
        stop={"training_iteration": iterations},
        checkpoint_freq=0,
        checkpoint_at_end=True,
        verbose=3,
        local_dir='./ray_dir',
    )

    # best_checkpoint = analysis.get_last_checkpoint(metric="episode_reward_mean", mode="max")
    return analysis.get_last_checkpoint()


def predict(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.Loss], trainer_type: Type[Trainable], checkpoint: str):
    env = init(game_type, losses)

    model = trainer_type(config=ppo_config())
    model.restore(checkpoint)

    obs = env.reset()
    actions = {agent: model.compute_single_action(obs[agent], unsquash_action=True, explore=False) for agent in pu.agents()}

    # output results
    print("Actions")
    print(f"action cont {actions[pu.cont()]}")
    print(f"action quiz {actions[pu.quiz()]}")
    print()

    obs, rewards, dones, infos = env.step(actions)

    print("Rewards")
    print(f"reward cont {rewards[pu.cont()]}")
    print(f"reward quiz {rewards[pu.quiz()]}")
    print()
