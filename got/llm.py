from abc import ABC, abstractmethod
import asyncio


class LLMAgentInterface(ABC):
    def __init__(self, name, model, apiKey):
        super().__init__()
        self.model = model
        self.apiKey = apiKey
        self.name = name

    @abstractmethod
    async def prompt(self, promptInput: str):
        pass


class OpenAIAgent(LLMAgentInterface):
    def __init__(self, name, model, apiKey):
        super().__init__(name, model, apiKey)

    async def prompt(self, promptInput: str):
        pass


class GoogleAgent(LLMAgentInterface):
    def __init__(self, name, model, apiKey):
        super().__init__(name, model, apiKey)

    async def prompt(self, promptInput: str):
        pass


class AnthropicAgent(LLMAgentInterface):
    def __init__(self, name, model, apiKey):
        super().__init__(name, model, apiKey)

    async def prompt(self, promptInput: str):
        pass
