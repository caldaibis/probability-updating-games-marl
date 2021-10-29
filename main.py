from __future__ import annotations

import gym

import probability_updating as pu
import probability_updating.game_samples.monty_hall as monty_hall

import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy, MultiInputPolicy


def simulate(g: pu.Game, l: pu.LossType, q: pu.PreStrategy, c: pu.PreStrategy):
    print(f"Game: {g.name}")

    g.loss_type = l
    print(f"Loss: {l}")
    print()

    g.quiz = q
    print(f"Quiz: {q.name}")

    g.cont = c
    print(f"Cont: {c.name}")

    print()
    print(f"CAR? {g.strategy.is_car()}")

    print(f"RCAR? {g.strategy.is_rcar()}")

    print()
    print(f"Expected loss: {g.loss.get_expected_loss()}")

    print(f"Expected entropy: {g.loss.get_expected_entropy()}")

    x_count, y_count, mean_loss, mean_entropy = g.simulate(100000)

    print()
    for x in x_count.keys():
        print(f"x{x.id}: {x_count[x]} times")

    print()
    for y in y_count.keys():
        print(f"y{y.id}: {y_count[y]} times")

    print()
    print(f"Mean loss: {mean_loss}")
    print(f"Mean entropy: {mean_entropy}")


def rl():
    env = gym.make('CartPole-v1')

    model = PPO(MlpPolicy, env, verbose=3)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done or info.get('is_success', False):
            obs = env.reset()


def ppo_chess():
    from pettingzoo.atari import pong_v2
    env = pong_v2.parallel_env(obs_type="grayscale_image", )
    env = ss.gym_vec_env_v0(env, num_envs=2, multiprocessing=True)

    model = PPO(MlpPolicy, env, verbose=3)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        action = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(action)
        env.render()
    env.close()


def test_pug():
    from pettingzoo.test import parallel_api_test
    env = pu.probability_updating_env.env(loss_type=pu.logarithmic())
    parallel_api_test(env, num_cycles=100)


def fixed_policy_pug():
    env = pu.probability_updating_env.env(loss_type=pu.logarithmic())

    for i in range(1):
        observations = env.reset()
        actions = {
            pu.quiz(): monty_hall.quiz_uniform(),
            pu.cont(): monty_hall.cont_min_loss_logarithmic()
        }
        observations, rewards, dones, infos = env.step(actions)

    env.close()


def ppo_pug():
    env = pu.probability_updating_env.env(loss_type=pu.logarithmic())
    env = ss.pad_action_space_v0(env)
    # env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    env = ss.concat_vec_envs_v0(env, 1, num_cpus=1, base_class='stable_baselines3')

    model = PPO(MlpPolicy, env, verbose=3)
    model.learn(total_timesteps=10000)

    for i in range(1):
        observations = env.reset()
        actions = {
            pu.quiz(): model.predict(observations[pu.quiz()], deterministic=True),
            pu.cont(): model.predict(observations[pu.cont()], deterministic=True)
        }
        observations, rewards, dones, infos = env.step(actions)

        print(rewards)

    env.close()


if __name__ == '__main__':
    # rl()
    # ppo_chess()
    # test_pug()
    # fixed_policy_pug()
    ppo_pug()

    # g = monty_hall.create_game()
    # simulate(g, pu.randomised_zero_one(), monty_hall.quiz_uniform(), monty_hall.cont_always_switch())
    # simulate(g, pu.brier(), monty_hall.quiz_uniform(), monty_hall.cont_min_loss_logarithmic())
    # simulate(g, pu.logarithmic(), monty_hall.quiz_uniform(), monty_hall.cont_min_loss_logarithmic())
