from __future__ import annotations

from typing import Callable

import probability_updating as pu
import probability_updating.game_samples.monty_hall as monty_hall

import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import inspect

import json


def simulate(game_fn: Callable[[pu.LossFunc, pu.LossFunc], pu.Game],
             loss_cont: pu.LossFunc | pu.Loss,
             loss_quiz: pu.LossFunc | pu.Loss,
             q: pu.PreStrategy,
             c: pu.PreStrategy):
    g = game_fn(loss_cont, loss_quiz)
    g.quiz = q
    g.cont = c

    print("BEGIN")
    print()

    print(f"Game: {g.name}")
    print()

    print(f"Loss (cont): {loss_cont if isinstance(loss_cont, pu.Loss) else inspect.getsource(loss_cont)}")
    print(f"Loss (quiz): {loss_quiz if isinstance(loss_quiz, pu.Loss) else inspect.getsource(loss_quiz)}")

    print(f"Strategy (cont): {c.name}")
    print(f"Strategy (quiz): {q.name}")

    print()
    print(f"CAR? {g.strategy.is_car()}")
    print(f"RCAR? {g.strategy.is_rcar()}")

    print()
    print(f"Expected loss (cont): {g.get_expected_loss(pu.cont())}")
    print(f"Expected entropy (cont): {g.get_expected_entropy(pu.cont())}")

    print()
    print(f"Expected loss (quiz): {g.get_expected_loss(pu.quiz())}")
    print(f"Expected entropy (quiz): {g.get_expected_entropy(pu.quiz())}")

    print()
    print("Running simulation...")
    sim = pu.SimulationWrapper(g)
    x_count, y_count, mean_loss, mean_entropy = sim.simulate(100000)

    print()
    for x in x_count.keys():
        print(f"x{x.id}: {x_count[x]} times")

    print()
    for y in y_count.keys():
        print(f"y{y.id}: {y_count[y]} times")

    print()
    print(f"Mean loss (cont): {mean_loss[pu.cont()]}")
    print(f"Mean loss (quiz): {mean_loss[pu.quiz()]}")

    print()
    print(f"Mean entropy (cont): {mean_entropy[pu.cont()]}")
    print(f"Mean entropy (quiz): {mean_entropy[pu.quiz()]}")

    print()
    print("END")


def monty_hall_test():
    from pettingzoo.test import parallel_api_test

    cont_loss = pu.Loss.randomised_zero_one
    quiz_loss = lambda c, o, x, y: -pu.randomised_zero_one(c, o, x, y)
    game = monty_hall.create_game(cont_loss, quiz_loss)

    env = pu.probability_updating_env.env(game=game)

    parallel_api_test(env, num_cycles=100)


def monty_hall_fixed():
    cont_loss = pu.Loss.randomised_zero_one
    quiz_loss = lambda c, o, x, y: -pu.randomised_zero_one(c, o, x, y)
    game = monty_hall.create_game(cont_loss, quiz_loss)

    env = pu.probability_updating_env.env(game=game)

    obs = env.reset()

    actions = {
        pu.cont(): monty_hall.cont_min_loss_logarithmic(),
        pu.quiz(): monty_hall.quiz_uniform()
    }

    obs, rewards, dones, infos = env.step(actions)

    print("Actions")
    print(pu.cont(), json.dumps(actions[pu.cont()].strategy, indent=2, default=str))
    print(pu.quiz(), json.dumps(actions[pu.quiz()].strategy, indent=2, default=str))
    print()

    print("Expected losses")
    print(pu.cont(), -rewards[pu.cont()])
    print(pu.quiz(), -rewards[pu.quiz()])
    print()

    print("Expected entropy", env.game.loss.get_expected_entropy())
    print()

    env.close()


def monty_hall_ppo_learn():
    cont_loss = pu.Loss.randomised_zero_one
    quiz_loss = lambda c, o, x, y: -pu.randomised_zero_one(c, o, x, y)
    game = monty_hall.create_game(cont_loss, quiz_loss)

    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    # TODO: Hyperparameter tuning!
    env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

    # TODO: Hyperparameter tuning!
    model = PPO(MlpPolicy, env, verbose=3)
    model.learn(total_timesteps=200000)
    model.save("ppo_g=monty_c=rand_q=-rand")

    env.close()


def monty_hall_ppo_predict():
    cont_loss = pu.Loss.randomised_zero_one
    quiz_loss = lambda c, o, x, y: -pu.randomised_zero_one(c, o, x, y)
    game = monty_hall.create_game(cont_loss, quiz_loss)

    env = pu.probability_updating_env.env(game=game)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    model = PPO.load("ppo_pug_competitive")

    obs = env.reset()

    actions = {
        0: model.predict(obs[0], deterministic=True)[0],
        1: model.predict(obs[1], deterministic=True)[0]
    }

    obs, rewards, dones, infos = env.step(actions)

    print("Actions (cont)", json.dumps(game.get_cont_readable(), indent=2, default=str))
    print("Actions (quiz)", json.dumps(game.get_quiz_readable(), indent=2, default=str))
    print()

    print("Expected loss (cont)", game.get_expected_loss(pu.cont()))
    print("Expected loss (quiz)", game.get_expected_loss(pu.quiz()))
    print()

    print("Expected entropy (cont)", game.get_expected_entropy(pu.cont()))
    print("Expected entropy (quiz)", game.get_expected_entropy(pu.quiz()))
    print()

    env.close()


if __name__ == '__main__':
    # simulate(monty_hall.create_game,
    #          pu.Loss.randomised_zero_one,
    #          lambda c, o, x, y: -pu.randomised_zero_one(c, o, x, y),
    #          monty_hall.quiz_uniform(),
    #          monty_hall.cont_always_switch())

    # monty_hall_ppo_learn()
    monty_hall_ppo_predict()
    # monty_hall_fixed()


    # m = pu.matrix_zero_one(len(monty_hall.marginal()))
    # simulate(monty_hall.create_game,
    #          lambda c, o, x, y: pu.randomised_matrix_loss(m, c, o, x, y),
    #          lambda c, o, x, y: -pu.randomised_matrix_loss(m, c, o, x, y),
    #          monty_hall.quiz_uniform(),
    #          monty_hall.cont_always_switch())

    # simulate(g, pu.brier(), monty_hall.quiz_uniform(), monty_hall.cont_min_loss_logarithmic())
    # simulate(g, pu.logarithmic(), monty_hall.quiz_uniform(), monty_hall.cont_min_loss_logarithmic())
