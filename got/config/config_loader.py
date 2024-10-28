from pathlib import Path
import yaml
import os
from typing import List
from got.llm import *

OPEN_AI = "OpenAI"
ANTHROPIC = "Anthropic"
GOOGLE = "Google"


class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(os.path.join(
            os.path.dirname(__file__), config_path))
        self._config = None
        self._load_config()

    def _load_config(self):
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Config file '{self.config_path}' not found.")

        with open(self.config_path, "r") as file:
            self._config = yaml.safe_load(file)
        print("aaaa")
        print(self._config)

    def get_agents(self) -> List[LLMAgentInterface]:
        agents = []
        llm_models = self._config["llm_models"]
        if OPEN_AI in llm_models:
            model = llm_models[OPEN_AI]["model"]
            apiKey = llm_models[OPEN_AI]["API_KEY"]
            agents.append(OpenAIAgent(OPEN_AI, model, apiKey))

        if ANTHROPIC in llm_models:
            model = llm_models[ANTHROPIC]["model"]
            apiKey = llm_models[ANTHROPIC]["API_KEY"]
            agents.append(AnthropicAgent(ANTHROPIC, model, apiKey))

        if GOOGLE in llm_models:
            model = llm_models[GOOGLE]["model"]
            apiKey = llm_models[GOOGLE]["API_KEY"]
            agents.append(GoogleAgent(GOOGLE, model, apiKey))

        return agents
