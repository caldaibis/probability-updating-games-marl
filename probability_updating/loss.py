from __future__ import annotations

import math
import probability_updating as pu


class Loss:
    game: pu.Game

    def __init__(self, game: pu.Game):
        self.game = game

    def get_expected_loss(self) -> float:
        loss: float = 0
        for x in self.game.outcomes:
            for y in self.game.messages:
                _l = self.game.marginal_outcome[x] * self.game.quiz[x][y] * self.get_loss(x, y)
                if not math.isnan(_l):
                    loss += _l

        return loss

    def get_loss(self, x: pu.Outcome, y: pu.Message) -> float:
        if self.game.loss_type == pu.randomised_zero_one():
            return self.randomised_zero_one(x, y)
        elif self.game.loss_type == pu.brier():
            return self.brier(x, y)
        elif self.game.loss_type == pu.logarithmic():
            return self.logarithmic(x, y)
        raise IndexError("not a valid loss type")

    def randomised_zero_one(self, x: pu.Outcome, y: pu.Message) -> float:
        return 1 - self.game.cont[y][x]

    def brier(self, x: pu.Outcome, y: pu.Message) -> float:
        loss: float = 0
        for x_ in self.game.outcomes:
            v = 1 if x == x_ else 0
            loss += math.pow(v - self.game.cont[y][x_], 2)

        return loss

    def logarithmic(self, x: pu.Outcome, y: pu.Message) -> float:
        return -pu.util.safe_log(self.game.cont[y][x])

    def get_expected_entropy(self) -> float:
        ent: float = 0
        for y in self.game.messages:
            e = self.game.marginal_message[y] * self.get_entropy(y)
            if not math.isnan(e):
                ent += e

        return ent

    def get_entropy(self, y: pu.Message) -> float:
        if self.game.loss_type is pu.randomised_zero_one():
            return self.randomised_zero_one_entropy(y)
        elif self.game.loss_type is pu.brier():
            return self.brier_entropy(y)
        elif self.game.loss_type is pu.logarithmic():
            return self.logarithmic_entropy(y)
        raise IndexError(f"not a valid loss type: {self.game.loss_type}")

    def randomised_zero_one_entropy(self, y: pu.Message) -> float:
        _max: float = 0
        for x in self.game.outcomes:
            if self.game.quiz_reverse[y][x] > _max:
                _max = self.game.quiz_reverse[y][x]

        return 1 - _max

    def brier_entropy(self, y: pu.Message) -> float:
        _sum: float = 0
        for x in self.game.outcomes:
            _sum += math.pow(self.game.quiz_reverse[y][x], 2)

        return 1 - _sum

    def logarithmic_entropy(self, y: pu.Message) -> float:
        _sum: float = 0
        for x in self.game.outcomes:
            e: float = -self.game.quiz_reverse[y][x] * pu.util.safe_log(self.game.quiz_reverse[y][x])
            if not math.isnan(e):
                _sum += e

        return _sum
