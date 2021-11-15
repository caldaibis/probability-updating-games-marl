from __future__ import annotations

import inspect
from typing import Type, Dict, Optional

import numpy as np

import probability_updating as pu
import probability_updating.games as games

from pathlib import Path

import models

model_path: Path = Path("saved_models")


def get_model_path(game: games.Game, losses: Dict[pu.Agent, pu.LossFunc | pu.Loss], model: Type[models.Model], total_timesteps: int, ext_name: Optional[str]):
    loss_str_c = losses[pu.cont()] if isinstance(losses[pu.cont()], pu.Loss) else ''
    loss_str_q = losses[pu.quiz()] if isinstance(losses[pu.quiz()], pu.Loss) else ''

    filename = f"c={loss_str_c}_q={loss_str_q}_tt={total_timesteps}_e={ext_name}"

    return f"{model_path}/{game.name()}/{model.name()}/{filename}"


def output_prediction(game: games.Game):
    import json

    print("Action (cont)", json.dumps(game.get_cont_readable(), indent=2, default=str))
    print("Action (quiz)", json.dumps(game.get_quiz_readable(), indent=2, default=str))
    print()

    print(f"Is CAR?  {game.strategy.is_car()}")
    print(f"Is RCAR? {game.strategy.is_rcar()}")
    print()

    print("Expected loss (cont)", game.get_expected_loss(pu.cont()))
    print("Expected loss (quiz)", game.get_expected_loss(pu.quiz()))
    print()

    print("Expected entropy (cont)", game.get_expected_entropy(pu.cont()))
    print("Expected entropy (quiz)", game.get_expected_entropy(pu.quiz()))
    print()


def manual(game, actions: Dict[pu.Agent, np.ndarray]):
    env = pu.probability_updating_env.env(game=game)
    env.reset()

    env.step(actions)

    output_prediction(game)


def learn(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.LossFunc | pu.Loss], model_type: Type[models.Model], total_timesteps: int, ext_name: Optional[str]):
    game = game_type(losses)

    model = model_type.create(game)
    model.learn(total_timesteps=total_timesteps)

    model.save(get_model_path(game, losses, model_type, total_timesteps, ext_name))


def predict(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.LossFunc | pu.Loss], model_type: Type[models.Model], total_timesteps: int, ext_name: Optional[str]):
    game = game_type(losses)

    model_type.predict(game, get_model_path(game, losses, model_type, total_timesteps, ext_name))

    output_prediction(game)


def test(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.LossFunc | pu.Loss], actions: Dict[pu.Agent, np.ndarray]):
    import simulation

    print("MANUAL TEST RUN BEGIN")
    print()
    print(f"Game: {game_type.name()}")

    game = game_type(losses)
    print(f"Action space (cont): {game.get_cont_action_space()}")
    print(f"Action space (quiz): {game.get_quiz_action_space()}")

    print(f"Loss function (cont): {losses[pu.cont()] if isinstance(losses[pu.cont()], pu.Loss) else inspect.getsource(losses[pu.cont()])}")
    print(f"Loss function (quiz): {losses[pu.quiz()] if isinstance(losses[pu.quiz()], pu.Loss) else inspect.getsource(losses[pu.quiz()])}")

    manual(game, actions)
    simulation.run(game, actions)

    print("MANUAL TEST RUN END")
    print()


def run():
    # MontyHall randomised    -> RCAR && Nash met PPO: total_timesteps=200000
    # MontyHall brier         -> RCAR && Nash met A2C: total_timesteps=500000
    # MontyHall logarithmic   -> RCAR && Nash met A2C: total_timesteps>1000000

    game_type = games.MontyHall
    losses = {
        pu.cont(): pu.Loss.logarithmic,
        pu.quiz(): lambda c, o, x, y: -pu.logarithmic(c, o, x, y)
    }
    actions = {
        pu.cont(): games.MontyHall.cont_min_loss_logarithmic(),
        pu.quiz(): games.MontyHall.quiz_uniform()
    }
    model_type = models.PPO
    total_timesteps = 1000000
    ext_name = ""

    # test(game_type, losses, actions)
    learn(game_type, losses, model_type, total_timesteps, ext_name)
    predict(game_type, losses, model_type, total_timesteps, ext_name)

    # import main_ray_llib
    # main_ray_llib.ray(game_type, losses)


if __name__ == '__main__':
    run()
