from __future__ import annotations

from pathlib import Path
from typing import Type, Dict, Optional

import probability_updating as pu
import probability_updating.games as games
import models


model_path: Path = Path("output_baselines")


def get_model_path(game: games.Game, losses: Dict[pu.Agent, pu.Loss], model: Type[models.Model], total_timesteps: int, ext_name: Optional[str]):
    filename = f"c={losses[pu.cont()].name}_q={losses[pu.quiz()].name}_tt={total_timesteps}_e={ext_name}"

    return f"{model_path}/{game.name()}/{model.name()}/{filename}"


def learn(game: games.Game, losses: Dict[pu.Agent, pu.Loss], model_type: Type[models.Model], total_timesteps: int, ext_name: Optional[str]):
    model = model_type.create(game)
    model.learn(total_timesteps=total_timesteps)

    model.save(get_model_path(game, losses, model_type, total_timesteps, ext_name))


def predict(game: games.Game, losses: Dict[pu.Agent, pu.Loss], model_type: Type[models.Model], total_timesteps: int, ext_name: Optional[str]):
    model_type.predict(game, get_model_path(game, losses, model_type, total_timesteps, ext_name))

    return game
