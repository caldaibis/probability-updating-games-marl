import probability_updating as pu
from pettingzoo.test import parallel_api_test


def monty_hall_test():
    game = pu.MontyHall.create(pu.Loss.randomised_zero_one, pu.Loss.randomised_zero_one)
    env = pu.probability_updating_env.env(game=game)

    parallel_api_test(env, num_cycles=100)
