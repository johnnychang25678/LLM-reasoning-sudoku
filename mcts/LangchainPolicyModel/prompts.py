from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.llms.base import BaseLLM

# Pydantic models for response validation
class SudokuState(BaseModel):
    rows: List[List[str]] = Field(description="2D list representing the Sudoku grid")

class GenerateActionsResponse(BaseModel):
    thoughts: str = Field(description="Reasoning about the possible next states")
    states: List[SudokuState] = Field(description="List of possible next states")

class TerminalStateResponse(BaseModel):
    thoughts: str = Field(description="Reasoning about whether the Sudoku is solved correctly")
    solved: bool = Field(description="Whether the Sudoku is solved correctly")

class TerminalStateRewardResponse(BaseModel):
    thoughts: str = Field(description="Reasoning about the state and reward")
    terminal: bool = Field(description="Whether the state is terminal")
    reward: int = Field(description="Reward value (-1, 0, or 1)")
    reason: str = Field(description="Explanation for the terminal state and reward")

# Prompt templates with escaped curly braces
gen_actions_prompt = PromptTemplate.from_template(
    """
    Given the following Sudoku state:
    {state}
    where * represents a cell to be filled in.

    Suggest the next possible states to solve this Sudoku by filling in exactly one * cell with a valid number. 
    You must strictly follow the Sudoku rule:
    - Each row must contain the numbers 1 through 3 exactly once.
    - Each column must contain the numbers 1 through 3 exactly once.
    - You are not allowed to modify any cell that already contains a number (i.e., non-* cells).

    **Reasoning Steps:**
    1. Identify all cells that are empty (i.e., contain '*').
    2. For each empty cell, determine the valid numbers that can be placed there based on the Sudoku rules.
    3. Generate a new state for each valid number by filling the empty cell with that number.

    Respond only with a valid JSON response. Do not include explanations outside the JSON. Explain your reasoning in the "thoughts" field.

    {format_instructions}
    """
)

is_terminal_and_give_reward_prompt = PromptTemplate.from_template(
    """
    Given the following Sudoku state:
    {state}
    where * represents a cell to be filled in.

    **Reasoning Steps:**
    1. Identify if the Sudoku state is terminal:
        - Check if there are any empty cells (i.e., '*'). If any are found, the state is not terminal.
    2. If the state is not terminal:
        - Explain why it is not terminal by identifying the empty cell(s).
        - Assign a reward of 0.
    3. If the state is terminal:
        - Verify if the Sudoku is solved correctly:
            - Check that each row contains the numbers 1 through 3 exactly once.
            - Check that each column contains the numbers 1 through 3 exactly once.
        - If solved correctly, assign a reward of +1.
        - If not solved correctly, assign a reward of -1.
        - Explain any rule violations.

    **Example Reasoning:**
    - Input state: [['1', '*', '3'], ['*', '2', '*'], ['*', '*', '*']]
    - Step 1: State is not terminal because cell (1, 2) is empty.
    - Step 2: Reward = 0.

    Respond only with a valid JSON response. Do not include explanations outside the JSON.

    {format_instructions}
    """
)

is_terminal_prompt = PromptTemplate.from_template(
    """
    Given the following Sudoku state:
    {state}
    where * represents a cell to be filled in.

    **Reasoning Steps:**
    1. Check if there are any empty cells ('*') in the grid.
    2. Verify if each row contains the numbers 1 through 3 exactly once.
    3. Verify if each column contains the numbers 1 through 3 exactly once.
    4. If any conditions fail, the Sudoku is not solved correctly.

    **Example Reasoning:**
    - Input state: [['1', '2', '3'], ['3', '1', '2'], ['2', '3', '1']]
    - Step 1: No empty cells found.
    - Step 2: Each row contains 1, 2, 3 exactly once.
    - Step 3: Each column contains 1, 2, 3 exactly once.
    - Step 4: Pass.

    Respond only with a valid JSON response. Do not include explanations outside the JSON.

    {format_instructions}
    """
)

# Chain functions incorporating escaped prompts
def get_actions_chain(llm: BaseLLM):
    parser = JsonOutputParser(pydantic_object=GenerateActionsResponse)
    prompt = PromptTemplate(
        template=gen_actions_prompt.template,
        input_variables=["state"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def get_terminal_chain(llm: BaseLLM):
    parser = JsonOutputParser(pydantic_object=TerminalStateResponse)
    prompt = PromptTemplate(
        template=is_terminal_prompt.template,
        input_variables=["state"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser

def get_terminal_reward_chain(llm: BaseLLM):
    parser = JsonOutputParser(pydantic_object=TerminalStateRewardResponse)
    prompt = PromptTemplate(
        template=is_terminal_and_give_reward_prompt.template,
        input_variables=["state"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser
