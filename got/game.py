
from abc import ABC, abstractmethod
from typing import List


class GameInterface(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def judge(self, responses: List[str]):
        # pick one best response
        pass


class SudokuGame(GameInterface):

    def judge(self, responses):
        pass
