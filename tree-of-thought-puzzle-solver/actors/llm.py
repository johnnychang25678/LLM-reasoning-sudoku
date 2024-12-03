# Import necessary modules
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from common.enums import ChatbotType
from common.config import Config
from common.hyperparams import HyperParams


class LLMAgent(object):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.chatbot = self._initialize_chatbot(config.chatbot_type)

    def compose_messages(self, roles, msg_content_list) -> object:
        if len(roles) != len(msg_content_list):
            raise ValueError(
                "Roles and message content lists must be of the same length.")
        msgs = []
        for role, content in zip(roles, msg_content_list):
            if role == "system":
                msgs.append(SystemMessage(content=content))
            elif role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown role: {role}")
        return msgs

    def get_reply(self, messages, temperature=None, max_tokens=None) -> str:
        return self.chatbot.get_reply(messages, temperature, max_tokens)

    def _initialize_chatbot(self, chatbot_type):
        if chatbot_type == ChatbotType.OpenAI:
            return OpenAIChatbot(
                self.config.openai_model,
                self.config.openai_version,
            )
        else:
            raise NotImplementedError(
                "Only OpenAI chatbot type is supported for now.")


class ChatbotBase(object):
    def __init__(self) -> None:
        pass

    def get_reply(self, messages, temperature=None, max_tokens=None) -> str:
        return ""


class OpenAIChatbot(ChatbotBase):
    def __init__(self, openai_model, openai_version) -> None:
        super().__init__()
        self.chat_model = AzureChatOpenAI(
            azure_deployment=openai_model,  # gpt-4o
            api_version=openai_version,  # 2024-02-15-preview
            temperature=HyperParams.DefaultTemperature,
            max_tokens=HyperParams.DefaultMaxTokens,
            # timeout=None,
            # max_retries=2,
        )

    def get_reply(self, messages, temperature=None, max_tokens=None) -> str:
        print("LLM Query:", messages)
        try:
            response = self.chat_model.invoke(messages)
            reply = response.content
            print("LLM Reply:", reply)
            print("")
            return reply
        except Exception as e:
            reply = "Failed to get LLM reply"
            print(reply)
            print(f"Error: {e}")
            return reply
