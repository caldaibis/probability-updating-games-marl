from __future__ import annotations

from abc import ABC, abstractmethod
import probability_updating as pu
from typing import List


class GameCreator(ABC):
    @classmethod
    def create(cls, loss_cont: pu.LossFunc | pu.Loss, loss_quiz: pu.LossFunc | pu.Loss) -> pu.Game:
        outcomes, messages = pu.GameCreator.create_structure(cls.marginal(), cls.message_structure())
        marginal_outcome = dict(zip(outcomes, cls.marginal()))

        return pu.Game(outcomes, messages, marginal_outcome, loss_cont, loss_quiz)

    @staticmethod
    @abstractmethod
    def name():
        pass

    @staticmethod
    @abstractmethod
    def marginal() -> List[float]:
        pass

    @staticmethod
    @abstractmethod
    def message_structure() -> List[List[int]]:
        pass

    @staticmethod
    def create_structure(marginal: List[float], messages: List[List[int]]) -> (List[pu.Outcome], List[pu.Message]):
        # create messages
        new_messages: List[pu.Message] = []
        for y in range(len(messages)):
            new_messages.append(pu.Message(y, []))

        # create outcomes
        new_outcomes: List[pu.Outcome] = []
        for x in range(len(marginal)):
            new_outcomes.append(pu.Outcome(x, []))

        # fill messages
        for y in range(len(messages)):
            for x in messages[y]:
                new_messages[y].outcomes.append(new_outcomes[x])

        # infer outcome structure from outcome count and message structure
        outcomes: List[List[int]] = []
        for x in range(len(marginal)):
            ys: List[int] = []
            for y in range(len(messages)):
                if x in messages[y]:
                    ys.append(y)
            outcomes.append(ys)

        # fill outcomes
        for x in range(len(outcomes)):
            for y in outcomes[x]:
                new_outcomes[x].messages.append(new_messages[y])

        return new_outcomes, new_messages

