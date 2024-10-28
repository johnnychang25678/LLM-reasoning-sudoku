
from abc import ABC, abstractmethod
from got.llm import LLMAgentInterface
from typing import List
from got.game import GameInterface
import asyncio


class SolveStrategy(ABC):
    def __init__(self, agents: List[LLMAgentInterface]):
        super().__init__()
        self.agents = agents

    async def asyncPrompt(self, promptInput: str) -> List[str]:
        """
        send prompt to all agents wait for responses
        """
        # Run each agent's prompt method asynchronously and collect responses
        tasks = [agent.prompt(promptInput) for agent in self.agents]
        # Waits for all tasks to complete
        responses = await asyncio.gather(*tasks)

        return responses

    @abstractmethod
    async def solve(self, game: GameInterface) -> bool:
        """
        use asyncPrompt to call all agents
        and use game.judge() to judge which one is best
        """
        pass


class ToTSolver(SolveStrategy):
    async def solve(self):
        pass


class XoTSolver(SolveStrategy):
    async def solve(self):
        pass


class GroupOfThoughts():
    def __init__(self, game: GameInterface, strategy: SolveStrategy):
        self.game = game
        self.strategy = strategy

    def set_strategy(self, strategy: SolveStrategy):
        self.strategy = strategy

    def run(self):
        self.strategy.solve(self.game)
