from __future__ import annotations

MONTY_HALL = "monty_hall"
FAIR_DIE = "fair_die"
EXAMPLE_C = "example_c"
EXAMPLE_D = "example_d"
EXAMPLE_E = "example_e"
EXAMPLE_F = "example_f"
EXAMPLE_G = "example_g"
EXAMPLE_H = "example_h"
COLOSSAL = "colossal"


import src.lib_pu.games.game
from src.lib_pu.games.game import Game

import src.lib_pu.games.monty_hall
import src.lib_pu.games.fair_die
import src.lib_pu.games.example_c
import src.lib_pu.games.example_d
import src.lib_pu.games.example_e
import src.lib_pu.games.example_f
import src.lib_pu.games.example_h
import src.lib_pu.games.example_g
import src.lib_pu.games.colossal


GAME_PRETTY_NAMES = {
    MONTY_HALL: "Monty Hall",
    FAIR_DIE  : "Fair Die",
    EXAMPLE_C : "Example C",
    EXAMPLE_D : "Example D",
    EXAMPLE_E : "Example E",
    EXAMPLE_F : "Example F",
    EXAMPLE_G : "Example G",
    EXAMPLE_H : "Example H",
    COLOSSAL: "Colossal",
}

GAMES = {
    MONTY_HALL: monty_hall.MontyHall,
    FAIR_DIE  : fair_die.FairDie,
    EXAMPLE_C : example_c.ExampleC,
    EXAMPLE_D : example_d.ExampleD,
    EXAMPLE_E : example_e.ExampleE,
    EXAMPLE_F : example_f.ExampleF,
    EXAMPLE_H : example_h.ExampleH,
    EXAMPLE_G : example_g.ExampleG,
    COLOSSAL : colossal.Colossal,
}


from src.lib_pu.games.monty_hall import MontyHall
from src.lib_pu.games.fair_die import FairDie
from src.lib_pu.games.example_c import ExampleC
from src.lib_pu.games.example_d import ExampleD
from src.lib_pu.games.example_e import ExampleE
from src.lib_pu.games.example_f import ExampleF
from src.lib_pu.games.example_h import ExampleH
from src.lib_pu.games.example_g import ExampleG
