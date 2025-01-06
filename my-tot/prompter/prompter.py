from abc import ABC, abstractmethod
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage


class Prompter(ABC):
    def __init__(self):
        super().__init__()

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
        # print("prompt: ", prompt)
        msgs = []
        msgs.append(HumanMessage(content=prompt))
        res = self.chatbot.invoke(msgs)
        reply = res.content
        print("llm reply: ", reply)
        return reply
