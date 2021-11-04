from __future__ import annotations

from typing import Callable

import probability_updating as pu
import probability_updating.game_samples.monty_hall as monty_hall

import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

import json


def simulate(game_fn: Callable[[pu.LossFunc, pu.LossFunc], pu.Game], loss_cont_fn: pu.LossFunc, loss_quiz_fn: pu.LossFunc, q: pu.PreStrategy, c: pu.PreStrategy):
    g = game_fn(loss_cont_fn, loss_quiz_fn)
    print(f"Game: {g.name}")
    print()

    print(f"Loss (contestant): {loss_cont_fn.__name__}")
    print(f"Loss (quizmaster): {loss_quiz_fn.__name__}")
    print()

    g.quiz = q
    print(f"Quiz: {q.name}")

    g.cont = c
    print(f"Cont: {c.name}")

    print()
    print(f"CAR? {g.strategy.is_car()}")

    print(f"RCAR? {g.strategy.is_rcar()}")

    print()
    print(f"Expected loss: {g.get_expected_loss()}")

    print(f"Expected entropy: {g.get_expected_entropy()}")

    sim = pu.SimulationWrapper(g)
    x_count, y_count, mean_loss, mean_entropy = sim.simulate(100000)

    print()
    for x in x_count.keys():
        print(f"x{x.id}: {x_count[x]} times")

    print()
    for y in y_count.keys():
        print(f"y{y.id}: {y_count[y]} times")

    print()
    print(f"Mean loss: {mean_loss}")
    print(f"Mean entropy: {mean_entropy}")


def monty_hall_test():
    from pettingzoo.test import parallel_api_test
    env = pu.probability_updating_env.env(game=monty_hall.create_game(), loss_type=pu.randomised_zero_one)
    parallel_api_test(env, num_cycles=100)


def monty_hall_fixed():
    env = pu.probability_updating_env.env(game=monty_hall.create_game(), loss_type=pu.brier)

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
    env = pu.probability_updating_env.env(game=monty_hall.create_game(), loss_type=pu.brier)
    env = ss.pad_action_space_v0(env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    # TODO: Hyperparameter tuning!
    env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

    # TODO: Hyperparameter tuning!
    model = PPO(MlpPolicy, env, verbose=3)
    model.learn(total_timesteps=100000)
    model.save("ppo_pug_competitive")

    env.close()


def monty_hall_ppo_predict():
    original_env = pu.probability_updating_env.env(game=monty_hall.create_game(), loss_type=pu.brier)
    env = ss.pad_action_space_v0(original_env)
    env = ss.agent_indicator_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)

    model = PPO.load("ppo_pug_competitive")

    obs = env.reset()

    actions = {
        0: model.predict(obs[0], deterministic=True)[0],
        1: model.predict(obs[1], deterministic=True)[0]
    }

    obs, rewards, dones, infos = env.step(actions)

    print("Actions")
    print(pu.cont(), json.dumps(original_env.game.strategy.to_pre_cont_strategy(actions[0]).strategy, indent=2, default=str))
    print(pu.quiz(), json.dumps(original_env.game.strategy.to_pre_quiz_strategy(actions[1]).strategy, indent=2, default=str))
    print()

    print("Losses")
    print(pu.cont(), -rewards[0])
    print(pu.quiz(), -rewards[1])
    print()

    print("Expected entropy", original_env.game.loss.get_expected_entropy())
    print()

    env.close()


if __name__ == '__main__':
    # monty_hall_ppo_learn()
    # monty_hall_ppo_predict()
    # monty_hall_fixed()

    simulate(monty_hall.create_game, pu.randomised_zero_one, lambda c, o, x, y: -pu.logarithmic(c, o, x, y), monty_hall.quiz_uniform(), monty_hall.cont_always_switch())
    # simulate(g, pu.brier(), monty_hall.quiz_uniform(), monty_hall.cont_min_loss_logarithmic())
    # simulate(g, pu.logarithmic(), monty_hall.quiz_uniform(), monty_hall.cont_min_loss_logarithmic())
