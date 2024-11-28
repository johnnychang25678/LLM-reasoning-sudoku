from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
import logging
import json

logger = logging.getLogger(__name__)

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
)

is_terminal_and_give_reward_prompt = PromptTemplate.from_template(
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

    Analyze the Sudoku state step by step. Do not write any code. Do not include generic explanations. Explain your reasoning.
    """
)

is_terminal_prompt = PromptTemplate.from_template(
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
)





class LangChainSudokuPolicyModel:
    def __init__(self, model_name="llama3.1:70b-instruct-q4_0"):
        """
        Initialize the model with LangChain's Ollama and LangSmith for observability.
        """
        self.llm = OllamaLLM(
            model=model_name,
        )
    
    def generate_actions_and_states(self, state):
        prompt_template = gen_actions_prompt
        chain = prompt_template | self.llm
        try:
            response = chain.invoke({"state": state})
            action_states = self._parse_actions_and_states(response, state)
            return action_states
        except Exception as e:
            logger.error(f"Error during generate_actions_and_states: {e}")
            return []
    
    def _parse_actions_and_states(self, response, cur_state):
        """
        Helper function to parse the actions and next states from the LLM response.
        """
        try:
            js = self._clean_json_helper(response)
            next_states = js["states"]
            action_states = []
            for next_state in next_states:
                if "rows" in next_state:
                    sudoku = next_state["rows"]
                    if self._check_state(cur_state, sudoku):
                        action = self._derive_action(cur_state, sudoku)
                        if action != "":
                            action_states.append((action, sudoku))
            return action_states
        except Exception as e:
            logger.error(f"Error parsing actions and states: {e}")
            return []

    def check_is_terminal(self, state):
        prompt_template = is_terminal_prompt
        chain = prompt_template | self.llm
        try:
            response = chain.invoke({"state": state})
            js = self._clean_json_helper(response)
            return js.get("solved", False)
        except Exception as e:
            logger.error(f"Error during check_is_terminal: {e}")
            return False

    def check_is_terminal_and_give_reward(self, state):
        prompt_template = is_terminal_and_give_reward_prompt
        chain = prompt_template | self.llm
        try:
            response = chain.invoke({"state": state})
            js = self._clean_json_helper(response)
            return js.get("terminal", False), js.get("reward", 0)
        except Exception as e:
            logger.error(f"Error during check_is_terminal_and_give_reward: {e}")
            return False, 0

    def _derive_action(self, cur_state, next_state):
        """
        Derives the action taken based on changes between cur_state and next_state.
        """
        actions = []
        for i in range(len(cur_state)):
            for j in range(len(cur_state[0])):
                before_value = cur_state[i][j]
                after_value = next_state[i][j]
                if before_value != after_value:
                    action_str = ""
                    if before_value == "*":
                        action_str = f"Filled cell ({i+1},{j+1}) with {after_value}"
                    if action_str:
                        actions.append(action_str)
        return "; ".join(actions) + "." if actions else "No changes made."

    def _check_state(self, cur_state, sudoku):
        """
        Check if the Sudoku grid is valid.
        """
        if not isinstance(sudoku, list):
            return False
        rows, cols = len(sudoku), len(sudoku[0])
        if rows != len(cur_state) or cols != len(cur_state[0]):
            return False
        for i in range(rows):
            for j in range(cols):
                ele = sudoku[i][j]
                if not isinstance(ele, str):
                    return False
                valid = (ele.isdigit() and int(ele) > 0) or ele == "*"
                if not valid:
                    return False
        return True

    def _clean_json_helper(self, response_text):
        """
        Cleans and parses JSON responses from the LLM.
        """
        try:
            cleaned_response = response_text.replace("\n", "").strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise e


if __name__ == "__main__":
    # policy_model = LangChainSudokuPolicyModel(
    #     model_name="llama3.2"
    # )
    
    test_cases = [
        [['1', '*', '*'], ['*', '1', '*'], ['*', '2', '*']],
        [['1', '2', '3'], ['3', '1', '2'], ['2', '3', '1']],
        [['1', '2', '3'], ['1', '2', '3'], ['2', '3', '1']],
    ]
    
    # for state in test_cases:
    #     terminal, reward = policy_model.check_is_terminal_and_give_reward(state)
    #     print(f"State: {state} -> Terminal: {terminal}, Reward: {reward}")

    model = OllamaLLM(model="llama3.2")
    prompt = is_terminal_and_give_reward_prompt
    chain = prompt | model
    for state in test_cases:
        print(chain.invoke({"state": state}))

