from __future__ import annotations

MONTY_HALL = "monty_hall"
FAIR_DIE = "fair_die"
EXAMPLE_C = "example_c"
EXAMPLE_D = "example_d"
EXAMPLE_E = "example_e"
EXAMPLE_F = "example_f"
EXAMPLE_H = "example_h"


import src.lib_pu.games.game
from src.lib_pu.games.game import Game

import src.lib_pu.games.monty_hall
import src.lib_pu.games.fair_die
import src.lib_pu.games.example_c
import src.lib_pu.games.example_d
import src.lib_pu.games.example_e
import src.lib_pu.games.example_f
import src.lib_pu.games.example_h


GAMES = {
    MONTY_HALL: monty_hall.MontyHall,
    FAIR_DIE  : fair_die.FairDie,
    EXAMPLE_C : example_c.ExampleC,
    EXAMPLE_D : example_d.ExampleD,
    EXAMPLE_E : example_e.ExampleE,
    EXAMPLE_F : example_f.ExampleF,
    EXAMPLE_H : example_h.ExampleH,
}

GAME_NAMES = {
    MONTY_HALL: monty_hall.MontyHall.name(),
    FAIR_DIE  : fair_die.FairDie.name(),
    EXAMPLE_C : example_c.ExampleC.name(),
    EXAMPLE_D : example_d.ExampleD.name(),
    EXAMPLE_E : example_e.ExampleE.name(),
    EXAMPLE_F : example_f.ExampleF.name(),
    EXAMPLE_H : example_h.ExampleH.name(),
}


from src.lib_pu.games.monty_hall import MontyHall
from src.lib_pu.games.fair_die import FairDie
from src.lib_pu.games.example_c import ExampleC
from src.lib_pu.games.example_d import ExampleD
from src.lib_pu.games.example_e import ExampleE
from src.lib_pu.games.example_f import ExampleF
from src.lib_pu.games.example_h import ExampleH

