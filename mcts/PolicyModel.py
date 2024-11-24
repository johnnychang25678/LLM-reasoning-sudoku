import requests
import logging
from mcts.mcts_prompts import gen_actions_prompt, gen_is_terminal_prompt, gen_is_terminal_and_give_reward_prompt
import json

logger = logging.getLogger(__name__)

class PolicyModel:
    def __init__(self):
        return NotImplemented

    def generate_actions_and_states(self, state):
        return NotImplemented

    def check_is_terminal(self, state):
        return NotImplemented

    def check_is_terminal_and_give_reward(self, state):
        return NotImplemented

class OllamaSudokuPolicyModel(PolicyModel):
    def __init__(self, api_endpoint: str = "http://localhost:11434/api", model_name: str = "llama3.1:70b-instruct-q4_0"):
        self.api_endpoint = api_endpoint
        self.model_name = model_name

    def generate_actions_and_states(self, state):
        prompt = gen_actions_prompt(state)
        # print(f"[DEBUG] Sending prompt to API: {prompt}")
        try:
            response_data = self.request_helper(prompt)
        except requests.RequestException as e:
            print(f"API call failed: {e}")
            return []

        action_states = []
        # response should be a json
        try:
            llm_res = response_data["response"]
            js = self.clean_json_helper(llm_res)
            # print("[DEBUG] json", js)
            logger.debug(f"[DEBUG] json: {js}")
            next_states = js["states"]
        except (KeyError, TypeError):
            print(f"Invalid API response: {response_data}")
            return []

        for next_state in next_states:
            if "rows" in next_state:
                sudoku = next_state["rows"]
                if self._check_state(state, sudoku):
                    action = self._derive_action(state, sudoku)
                    if action != "":
                        action_states.append((action, sudoku))
        # [("action...", "new_state")]
        return action_states
    
    def _check_state(self, cur_state, sudoku):
        """check if sudoku size is same as cur_state and if sudoku cells only consists of * or int"""
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

    def _derive_action(self, cur_state, next_state):
        actions = []
        for i in range(len(cur_state)):
            for j in range(len(cur_state[0])):
                before_value = cur_state[i][j]
                after_value = next_state[i][j]
                if before_value != after_value:
                    # Record the change
                    action_str = ""
                    if before_value == "*":
                        action_str = f"Filled cell ({i+1},{j+1}) with {after_value}"
                    else:
                        # if empty or change cell, it violates sudoku rule, do we record it or not? TODO
                        print("invalid action, throw away")
                        return ""
                    # elif after_value == "*":
                    #     action_str = f"Emptied cell ({i+1},{j+1}) which was {before_value}"

                    # else:
                    #     action_str = f"Changed cell ({i+1},{j+1}) from {before_value} to {after_value}"
                    if action_str != "":
                        actions.append(action_str)
        # Combine all actions into a single description
        if actions:
            action_description = "; ".join(actions) + "."
        else:
            action_description = "No changes made."

        out = f"Previous state: {cur_state}. Action taken: {action_description}"
        return out

    def check_is_terminal(self, state):
        prompt = gen_is_terminal_prompt(state)
        try:
            response_data = self.request_helper(prompt)
        except requests.RequestException as e:
            print(f"API call failed: {e}")
            return False

        try:
            llm_res = response_data["response"]
            js = self.clean_json_helper(llm_res)
            print("[DEBUG] json", js)
            return js["solved"]
        except KeyError:
            print(f"Invalid API response: {response_data}")
            return False

    def request_helper(self, prompt):
        """helper function to call api endpoint"""
        try:
            response = requests.post(
                f"{self.api_endpoint}/generate", json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                })
            response.raise_for_status()
            response_data = response.json()
            # print(f"[DEBUG] API Response: {response_data}")
            return response_data
        except requests.RequestException as e:
            raise e

    def clean_json_helper(self, text: str):
        """helper function to clean json respone from llama"""
        cleaned_response = text.replace("\n", "").strip()
        try:
            parsed_data = json.loads(cleaned_response)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")

    def check_is_terminal_and_give_reward(self, state):
        '''
        Check if the state is terminal and if so, give corresponding reward
        '''
        prompt = gen_is_terminal_and_give_reward_prompt(state)
        try:
            response_data = self.request_helper(prompt)
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            return False, 0
        
        try:
            llm_res = response_data["response"]
            js = self.clean_json_helper(llm_res)
            return js["terminal"], js["reward"]
        except KeyError:
            logger.error(f"Invalid API response: {response_data}")
            return False, 0

if __name__ == "__main__":
    policy_model = OllamaSudokuPolicyModel()
    print(policy_model.check_is_terminal_and_give_reward([['1', '*', '*'], ['*', '1', '*'], ['*', '2', '*']]))
    print(policy_model.check_is_terminal_and_give_reward([['1', '2', '3'], ['3', '1', '2'], ['2', '3', '1']]))
    print(policy_model.check_is_terminal_and_give_reward([['1', '2', '3'], ['1', '2', '3'], ['2', '3', '1']]))
