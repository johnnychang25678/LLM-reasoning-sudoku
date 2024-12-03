from langchain_ollama.llms import OllamaLLM
import logging

from mcts.LangchainPolicyModel.LangchainCallbackHandler import LoggingCallbackHandler
from mcts.LangchainPolicyModel.chain import (
    get_actions_chain,
    get_terminal_chain,
    get_terminal_reward_chain
)

from mcts.LangchainPolicyModel.jsonParser import parse_json

logger = logging.getLogger(__name__)

class LangchainOllamaSudokuPolicyModel:
    def __init__(self, model_name="llama3.1:70b-instruct-q4_0", callback_handler=None):
        """
        Initialize the model with LangChain's Ollama and LangSmith for observability.
        """
        if callback_handler:
            self.llm = OllamaLLM(
                model=model_name,
                callbacks=[callback_handler]
            )
            logger.info(f"Using callback handler: {callback_handler}")
        else:
            self.llm = OllamaLLM(
                model=model_name,
            )
            logger.info(f"Not using callback handler")
        
        # Initialize chains with existing prompts
        self.actions_chain = get_actions_chain(self.llm, parse_json)
        self.terminal_chain = get_terminal_chain(self.llm, parse_json)
        self.terminal_reward_chain = get_terminal_reward_chain(self.llm, parse_json)
    
    def generate_actions_and_states(self, state):
        try:
            response = self.actions_chain.invoke({"state": state})
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
            next_states = response["states"]
            action_states = []
            for next_state in next_states:
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
        try:
            response = self.terminal_chain.invoke({"state": state})
            return response["solved"]
        except Exception as e:
            logger.error(f"Error during check_is_terminal: {e}")
            return False
    
    def check_is_terminal_and_give_reward(self, state):
        try:
            response = self.terminal_reward_chain.invoke({"state": state})
            return response["terminal"], response["reward"]
        except Exception as e:
            logger.error(f"Error during check_is_terminal_and_give_reward: {e}")
            return False, 0
    
    def _derive_action(self, cur_state, next_state):
        """
        Derives the action taken based on changes between cur_state and next_state.
        """
        # actions = []
        # for i in range(len(cur_state)):
        #     for j in range(len(cur_state[0])):
        #         before_value = cur_state[i][j]
        #         after_value = next_state[i][j]
        #         if before_value != after_value:
        #             action_str = ""
        #             if before_value == "*":
        #                 action_str = f"Filled cell ({i+1},{j+1}) with {after_value}"
        #             if action_str:
        #                 actions.append(action_str)
        # return "; ".join(actions) + "." if actions else "No changes made."
        return "current state: " + str(cur_state) + " -> next state: " + str(next_state)
    
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

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    policy_model = LangChainSudokuPolicyModel(
        model_name="llama3.1:70b-instruct-q2_K",
        callback_handler=LoggingCallbackHandler()
    )
    
    test_cases = [
        [['1', '*', '*'], ['*', '1', '*'], ['*', '2', '*']],
        # [['1', '2', '3'], ['3', '1', '2'], ['2', '3', '1']],
        # [['1', '2', '3'], ['1', '2', '3'], ['2', '3', '1']],
    ]
    
    # verify the reward and terminal check
    # for state in test_cases:
    #     terminal, reward = policy_model.check_is_terminal_and_give_reward(state)
    #     print(f"State: {state} -> Terminal: {terminal}, Reward: {reward}")

    # verify the action generation
    for state in test_cases:
        actions = policy_model.generate_actions_and_states(state)
        print(f"State: {state} -> Actions: {actions}")

    # model = OllamaLLM(model="llama3.2")
    # prompt = is_terminal_and_give_reward_prompt
    # chain = prompt | model
    # for state in test_cases:
    #     print(chain.invoke({"state": state}))

