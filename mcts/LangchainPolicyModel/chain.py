from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.llms.base import BaseLLM
from mcts.LangchainPolicyModel.ollamaPrompts import gen_actions_prompt, is_terminal_prompt, is_terminal_and_give_reward_prompt
from mcts.LangchainPolicyModel.jsonParser import parse_json

def get_actions_chain(llm: BaseLLM, parser):
    prompt = PromptTemplate(
        template=gen_actions_prompt,
        input_variables=["state"],
    )
    return prompt | llm | parser

def get_terminal_chain(llm: BaseLLM, parser):
    prompt = PromptTemplate(
        template=is_terminal_prompt,
        input_variables=["state"],
    )
    return prompt | llm | parser

def get_terminal_reward_chain(llm: BaseLLM, parser):
    prompt = PromptTemplate(
        template=is_terminal_and_give_reward_prompt,
        input_variables=["state"],
    )
    return prompt | llm | parser
