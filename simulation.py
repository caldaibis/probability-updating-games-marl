from __future__ import annotations

import probability_updating as pu

from typing import Callable

import inspect


def run(game_fn: Callable[[pu.LossFunc, pu.LossFunc], pu.Game],
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
