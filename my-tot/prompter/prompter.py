from abc import ABC, abstractmethod


class Prompter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def prompt(prompt: str):
        pass


class AzureOpenAIPrompter(Prompter):
    def __init__(self, model, model_version, temperature, max_tokens, leaf_count):
        super().__init__()
        self.model = model
        self.model_version = model_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.leaf_count = leaf_count

    def prompt(prompt):
        # TODO
        pass
