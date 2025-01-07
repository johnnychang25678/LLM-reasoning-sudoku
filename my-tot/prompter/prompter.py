from abc import ABC, abstractmethod
from observer.observer import Observer
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage


class Prompter(ABC):
    def __init__(self):
        self.observers: List[Observer] = []
        self.prompt_count = 0
        self.input_token_count = 0

    def add_observer(self, observer: Observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.notify(message)

    def get_prompt_count(self):
        return self.prompt_count

    def get_input_token_count(self):
        return self.input_token_count

    @abstractmethod
    def prompt(prompt: str):
        pass


class AzureOpenAIPrompter(Prompter):
    def __init__(self, model, model_version, temperature, max_tokens, leaf_count):
        super().__init__()
        self.leaf_count = leaf_count
        self.chatbot = AzureChatOpenAI(
            azure_deployment=model,  # gpt-4o
            api_version=model_version,  # 2024-02-15-preview
            temperature=temperature,
            max_tokens=max_tokens,
            # timeout=None,
            # max_retries=2,
        )

    def prompt(self, prompt):
        msgs = []
        msgs.append(HumanMessage(content=prompt))
        res = self.chatbot.invoke(msgs)
        reply = res.content
        token_count = res.usage_metadata["input_tokens"]
        self.notify_observers(f"llm reply: {reply}")
        self.prompt_count += 1
        self.input_token_count += token_count
        return reply
