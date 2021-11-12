from __future__ import annotations

from typing import Type

import probability_updating as pu

import supersuit as ss
from pathlib import Path

import models as models

import simulation


model_path: Path = Path("saved_models")

cont_loss = pu.Loss.randomised_zero_one
quiz_loss = lambda c, o, x, y: -pu.randomised_zero_one(c, o, x, y)


def output_prediction(game: pu.Game):
    import json

    print("Actions (cont)", json.dumps(game.get_cont_readable(), indent=2, default=str))
    print("Actions (quiz)", json.dumps(game.get_quiz_readable(), indent=2, default=str))
    print()

    print("Expected loss (cont)", game.get_expected_loss(pu.cont()))
    print("Expected loss (quiz)", game.get_expected_loss(pu.quiz()))
    print()

    print("Expected entropy (cont)", game.get_expected_entropy(pu.cont()))
    print("Expected entropy (quiz)", game.get_expected_entropy(pu.quiz()))
    print()


def monty_hall_fixed():
    game = pu.MontyHall.create(cont_loss, quiz_loss)
    env = pu.probability_updating_env.env(game=game)

    obs = env.reset()

    actions = {
        pu.cont(): pu.MontyHall.cont_always_switch(),
        pu.quiz(): pu.MontyHall.quiz_uniform()
    }

    obs, rewards, dones, infos = env.step(actions)

    env.close()
    output_prediction(game)


def monty_hall_ppo_learn():
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy

    game = pu.MontyHall.create(cont_loss, quiz_loss)
    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    # TODO: Hyperparameter tuning!
    env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

    model = PPO(MlpPolicy, env, verbose=3)

    # TODO: Hyperparameter tuning!
    model.learn(total_timesteps=25000)

    model.save(model_path / model_name)
    env.close()


def monty_hall_ppo_predict():
    from stable_baselines3 import PPO

    game = pu.MontyHall.create(cont_loss, quiz_loss)
    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    model = PPO.load(model_path / model_name)

    obs = env.reset()

    actions = {
        0: model.predict(obs[0], deterministic=True)[0],
        1: model.predict(obs[1], deterministic=True)[0]
    }

    obs, rewards, dones, infos = env.step(actions)

    env.close()
    output_prediction(game)


def monty_hall_td3_learn():
    from stable_baselines3 import TD3
    from stable_baselines3.td3 import MlpPolicy

    game = pu.MontyHall.create(cont_loss, quiz_loss)
    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    # TODO: Hyperparameter tuning!
    env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

    model = TD3(MlpPolicy, env, verbose=3)

    # TODO: Hyperparameter tuning!
    model.learn(total_timesteps=25000)

    model.save(model_path / model_name)
    env.close()


def monty_hall_td3_predict():
    from stable_baselines3 import TD3

    game = pu.MontyHall.create(cont_loss, quiz_loss)
    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    model = TD3.load(model_path / model_name)

    obs = env.reset()

    actions = {
        0: model.predict(obs[0], deterministic=True)[0],
        1: model.predict(obs[1], deterministic=True)[0]
    }

    obs, rewards, dones, infos = env.step(actions)

    env.close()
    output_prediction(game)


def monty_hall_a2c_learn():
    from stable_baselines3 import A2C
    from stable_baselines3.a2c import MlpPolicy

    game = pu.MontyHall.create(cont_loss, quiz_loss)
    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    # TODO: Hyperparameter tuning!
    env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

    model = A2C(MlpPolicy, env, verbose=3)

    # TODO: Hyperparameter tuning!
    model.learn(total_timesteps=25000)

    model.save(model_path / model_name)
    env.close()


def monty_hall_a2c_predict():
    from stable_baselines3 import A2C

    game = pu.MontyHall.create(cont_loss, quiz_loss)
    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    model = A2C.load(model_path / model_name)

    obs = env.reset()

    actions = {
        0: model.predict(obs[0], deterministic=True)[0],
        1: model.predict(obs[1], deterministic=True)[0]
    }

    obs, rewards, dones, infos = env.step(actions)

    env.close()
    output_prediction(game)


def learn(game_creator: Type[pu.GameCreator], loss_cont: pu.LossFunc | pu.Loss, loss_quiz: pu.LossFunc | pu.Loss, model_creator: Type[mc.Model]):
    game = game_creator.create(loss_cont, loss_quiz)

    model = model_creator.create(game)
    model.learn(total_timesteps=25000)

    model.save(f"{model_path}/{game_creator.name()}/{model_creator.name()}/{loss_cont if isinstance(loss_cont, pu.Loss) else ''}_{loss_quiz if isinstance(loss_quiz, pu.Loss) else ''}")


def predict(game_creator: Type[pu.GameCreator], loss_cont: pu.LossFunc | pu.Loss, loss_quiz: pu.LossFunc | pu.Loss, model_creator: Type[mc.Model]):
    game = game_creator.create(loss_cont, loss_quiz)

    model_creator.predict(game, f"{model_path}/{game_creator.name()}/{model_creator.name()}/{loss_cont if isinstance(loss_cont, pu.Loss) else ''}_{loss_quiz if isinstance(loss_quiz, pu.Loss) else ''}")

    output_prediction(game)



if __name__ == '__main__':
    # simulation.run(
    #     pu.MontyHall,
    #     cont_loss,
    #     quiz_loss,
    #     pu.MontyHall.quiz_always_y2(),
    #     pu.MontyHall.cont_always_switch()
    # )

    # monty_hall_fixed()

    # monty_hall_ppo_learn()
    # monty_hall_ppo_predict()

    # monty_hall_td3_learn()
    # monty_hall_td3_predict()

    # monty_hall_a2c_learn()
    # monty_hall_a2c_predict()

    c_loss = pu.Loss.randomised_zero_one
    q_loss = lambda c, o, x, y: -pu.randomised_zero_one(c, o, x, y)

    learn(pu.FairDie, c_loss, q_loss, mc.A2C)
    predict(pu.MontyHall, c_loss, q_loss, mc.A2C)

    # m = pu.matrix_zero_one(len(monty_hall.marginal()))
    # simulation.run(monty_hall.create_game,
    #          lambda c, o, x, y: pu.randomised_matrix_loss(m, c, o, x, y),
    #          lambda c, o, x, y: -pu.randomised_matrix_loss(m, c, o, x, y),
    #          monty_hall.quiz_uniform(),
    #          monty_hall.cont_always_switch())

    # simulation.run(g, pu.brier(), monty_hall.quiz_uniform(), monty_hall.cont_min_loss_logarithmic())
    # simulation.run(g, pu.logarithmic(), monty_hall.quiz_uniform(), monty_hall.cont_min_loss_logarithmic())
