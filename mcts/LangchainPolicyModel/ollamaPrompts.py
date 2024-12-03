from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.llms.base import BaseLLM

# Pydantic models for response validation

class SudokuState(BaseModel):
    rows: List[List[str]] = Field(
        description="2D list representing the Sudoku grid. Each string must be '1', '2', '3', or '*'."
    )

class GenerateActionsResponse(BaseModel):
    states: List[SudokuState] = Field(
        description=(
            "List of possible next states. Each state must be represented as an object with a single key 'rows', "
            "which contains a 2D list of strings representing the Sudoku grid."
        )
    )

class TerminalStateResponse(BaseModel):
    solved: bool = Field(
        description=(
            "Indicates whether the Sudoku is solved correctly. Must be 'true' if each row and column contains the "
            "numbers 1 through 3 exactly once and there are no empty cells ('*'). Otherwise, 'false'."
        )
    )

class TerminalStateRewardResponse(BaseModel):
    terminal: bool = Field(
        description=(
            "Indicates whether the state is terminal. Must be 'true' if there are no empty cells ('*'), otherwise 'false'."
        )
    )
    reward: int = Field(
        description=(
            "Reward value. Must be 1 if the terminal state is solved correctly, -1 if it is terminal but not solved "
            "correctly, and 0 if the state is not terminal."
        )
    )
    reason: str = Field(
        description=(
            "Explanation for the terminal state and reward. If not terminal, explain which cells are empty. "
            "If terminal, explain correctness or rule violations."
        )
    )
    
gen_actions_prompt = \
"""
Given the following Sudoku state:
{state}
where * represents a cell to be filled in.

Suggest the next possible states to solve this Sudoku by filling in exactly one * cell with a valid number. 
You must strictly follow the Sudoku rule:
- Each row must contain the numbers 1 through 3 exactly once.
- Each column must contain the numbers 1 through 3 exactly once.
- You are not allowed to modify any cell that already contains a number (i.e., non-* cells).

**Schema Requirements:**
- Your response must be valid JSON.
- The JSON object must have a single key "states".
- The value of "states" must be a list.
- Each item in the list must be an object with a single key "rows".
- The value of "rows" must be a 2D list of strings, where each string is either:
    - A number between "1" and "3" (inclusive).
    - The string "*", representing an empty cell.
- Each suggested state must differ from the original state by exactly one cell.

Example:
If the input state is:
[
    ["1", "*", "3"],
    ["*", "2", "*"],
    ["*", "*", "*"]
]
A valid response could be:
{{
    "states": [
        {{
            "rows": [
                ["1", "2", "3"],
                ["*", "2", "*"],
                ["*", "*", "*"]
            ]
        }},
        {{
            "rows": [
                ["1", "*", "3"],
                ["3", "2", "*"],
                ["*", "*", "*"]
            ]
        }}
    ]
}}

Respond only with valid JSON following the schema. Do not write an introduction, summary, or explanations.

"""

is_terminal_and_give_reward_prompt = \
"""
Given the following Sudoku state:
{state}
where * represents a cell to be filled in.

**Tasks:**
1. Determine if the Sudoku state is a terminal state. A terminal state has:
    - No empty cells (i.e., no * present in the grid).
2. If the state is not terminal:
    - Provide a reason explaining why it is not terminal (e.g., "Cell at (row, column) is empty.").
    - Assign a reward of 0.
3. If the state is terminal:
    - Check if the Sudoku is solved correctly. A Sudoku is solved correctly if:
        - Each row contains the numbers 1 through 3 exactly once.
        - Each column contains the numbers 1 through 3 exactly once.
    - Assign a reward of +1 if the Sudoku is solved correctly and -1 if it is not.
    - Provide a reason, even if terminal, explaining correctness or violation of the rules.

**Schema Requirements:**
- Your response must be valid JSON.
- The JSON object must contain three keys: "terminal", "reward", and "reason".
- The value of "terminal" must be a boolean:
    - `true` if the state is terminal (no empty cells).
    - `false` if the state is not terminal (contains empty cells).
- The value of "reward" must be an integer:
    - `1` if the state is terminal and correctly solved.
    - `-1` if the state is terminal but not solved correctly.
    - `0` if the state is not terminal.
- The value of "reason" must be a string explaining why the Sudoku is or is not terminal and, if terminal, why it is correct or incorrect.

**Examples:**
- Terminal and Correct Sudoku: 
    [['1', '2', '3'], ['3', '1', '2'], ['2', '3', '1']] -> 
    {{
        "terminal": true,
        "reward": 1,
        "reason": "The Sudoku is solved correctly with no empty cells and no rule violations."
    }}
- Terminal and Incorrect Sudoku: 
    [['1', '2', '3'], ['3', '1', '2'], ['2', '3', '2']] -> 
    {{
        "terminal": true,
        "reward": -1,
        "reason": "The Sudoku is incorrect because the number 2 is repeated in the last column."
    }}
- Non-terminal Sudoku: 
    [['1', '*', '3'], ['3', '1', '2'], ['2', '3', '1']] -> 
    {{
        "terminal": false,
        "reward": 0,
        "reason": "Cell at (1, 2) is empty."
    }}

Respond only with valid JSON following the schema. Example response:
{{
    "terminal": false,
    "reward": 0,
    "reason": "Cell at (1, 2) is empty."
}}

Do not write an introduction, summary, or explanation.
"""

is_terminal_prompt = \
"""
Given the following Sudoku state:
{state}
where * represents a cell to be filled in.

Determine if the Sudoku is solved correctly. A Sudoku is considered solved if:
- Each row contains the numbers 1 through 3 exactly once.
- Each column contains the numbers 1 through 3 exactly once.
- There are no empty cells (i.e., no * present in the grid).

**Schema Requirements:**
- Your response must be valid JSON.
- The JSON object must contain a single key "solved".
- The value of "solved" must be a boolean:
    - `true` if the Sudoku is solved correctly.
    - `false` otherwise.

**Examples:**
- Valid Sudoku: [['1', '2', '3'], ['3', '1', '2'], ['2', '3', '1']] -> {{"solved": true}}
- Invalid Sudoku: [['1', '*', '3'], ['3', '1', '2'], ['2', '3', '1']] -> {{"solved": false}}

Ensure your response is strictly based on the conditions provided. Do not assume correctness unless all conditions are explicitly met.

Respond only with valid JSON following the schema. Example response:
{{
    "solved": false
}}

Do not write an introduction, summary, or explanation.
"""

# Chain functions incorporating escaped prompts
