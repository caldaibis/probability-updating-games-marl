from __future__ import annotations

import inspect
from typing import Type, Dict

import probability_updating as pu
import probability_updating.games as games

from pathlib import Path

import simulation
import models

model_path: Path = Path("saved_models")


def get_model_path(game: games.Game, losses: Dict[pu.Agent, pu.LossFunc | pu.Loss], model: Type[models.Model]):
    return f"{model_path}/{game.name()}/{model.name()}/{losses[pu.cont()] if isinstance(losses[pu.cont()], pu.Loss) else ''}_{losses[pu.quiz()] if isinstance(losses[pu.quiz()], pu.Loss) else ''}"


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


def manual(game, actions: Dict[pu.Agent, pu.StrategyWrapper]):
    env = pu.probability_updating_env.env(game=game)
    env.reset()

    env.step(actions)

    output_prediction(game)


def learn(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.LossFunc | pu.Loss], model_type: Type[models.Model], total_timesteps: int):
    game = game_type(losses)

    model = model_type.create(game)
    model.learn(total_timesteps=total_timesteps)

    model.save(get_model_path(game, losses, model_type))


def predict(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.LossFunc | pu.Loss], model_type: Type[models.Model]):
    game = game_type(losses)

    model_type.predict(game, get_model_path(game, losses, model_type))

    output_prediction(game)


def test(game_type: Type[games.Game], losses: Dict[pu.Agent, pu.LossFunc | pu.Loss], actions: Dict[pu.Agent, pu.StrategyWrapper]):
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
    # randomised lijkt het erg goed te doen, voor MontyHall en FairDie met A2C of PPO!
    # logarithmic lijkt niet te convergeren naar Nash met A2C of PPO...
    # brier lijkt niet te convergeren naar Nash met PPO of A2C...

    game_type = games.MontyHall
    losses = {
        pu.cont(): pu.Loss.logarithmic,
        pu.quiz(): lambda c, o, x, y: -pu.logarithmic(c, o, x, y)
    }
    actions = {
        pu.cont(): games.MontyHall.cont_min_loss_logarithmic(),
        pu.quiz(): games.MontyHall.quiz_uniform()
    }
    model_type = models.A2C
    total_timesteps = 10000

    # test(game_type, losses, actions)
    learn(game_type, losses, model_type, total_timesteps)
    predict(game_type, losses, model_type)


if __name__ == '__main__':
    run()
