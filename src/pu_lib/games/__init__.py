from __future__ import annotations

MONTY_HALL = "monty_hall"
FAIR_DIE = "fair_die"
EXAMPLE_C = "example_c"
EXAMPLE_D = "example_d"
EXAMPLE_E = "example_e"
EXAMPLE_F = "example_f"
EXAMPLE_H = "example_h"


import src.pu_lib.games.game
from src.pu_lib.games.game import Game

import src.pu_lib.games.monty_hall
import src.pu_lib.games.fair_die
import src.pu_lib.games.example_c
import src.pu_lib.games.example_d
import src.pu_lib.games.example_e
import src.pu_lib.games.example_f
import src.pu_lib.games.example_h


GAMES = {
    MONTY_HALL: monty_hall.MontyHall,
    FAIR_DIE  : fair_die.FairDie,
    EXAMPLE_C : example_c.ExampleC,
    EXAMPLE_D : example_d.ExampleD,
    EXAMPLE_E : example_e.ExampleE,
    EXAMPLE_F : example_f.ExampleF,
    EXAMPLE_H : example_h.ExampleH,
}


from src.pu_lib.games.monty_hall import MontyHall
from src.pu_lib.games.fair_die import FairDie
from src.pu_lib.games.example_c import ExampleC
from src.pu_lib.games.example_d import ExampleD
from src.pu_lib.games.example_e import ExampleE
from src.pu_lib.games.example_f import ExampleF
from src.pu_lib.games.example_h import ExampleH

