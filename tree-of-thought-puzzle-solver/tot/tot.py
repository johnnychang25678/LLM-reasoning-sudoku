import json
import common.consts as consts
import common.utils as utils
from common.enums import *
from common.hyperparams import HyperParams
from actors.state import SudokuStateManager
from actors.llm import LLMAgent
from actors.parser import LLMReplyParserForSudoku
from actors.prompter import SudokuPrompter
# from ValueModel.Load_Inference import RegressionPredictor
from ValueModel.valuemodelinference_lstm import LSTMPredictor

import numpy as np


def numpy_matrices_to_custom_strings(matrices):
    """
    Converts a list of NumPy matrices into strings formatted as:
    [['3', '1', '2'], ['1', '2', '*'], ...].
    """
    result = []
    for matrix in matrices:
        # Convert to a NumPy array if it's not already
        matrix = np.array(matrix)

        # Squeeze to remove any extra dimensions
        matrix = np.squeeze(matrix)

        # Now we expect matrix to be 2D: (rows, cols)
        # If there's still a nested structure, flatten each row.
        rows_str = []
        for row in matrix:
            # Ensure row is a 1D array
            row = np.array(row).flatten()
            # Convert each element to a quoted string
            row_str = ", ".join(f"'{str(item)}'" for item in row)
            rows_str.append(f"[{row_str}]")

        # Join all rows with ", " and wrap in outer brackets
        matrix_str = "[" + ", ".join(rows_str) + "]"
        result.append(matrix_str)

    return result


# Example usage:
# matrices = [
#     np.array([[3, 1, 2], [1, 2, '*'], ['*', 3, 4]]),
#     np.array([[10, 11], [12, '*']]),
#     np.array([[1]])
# ]


class TreeOfThought(object):

    def __init__(self, config, prompt_type: PromptGenType) -> None:
        self.config = config
        self.llm_agent = LLMAgent(config)
        self.prompt_type = prompt_type

    def run(self, user_input):
        max_num_rounds = HyperParams.MaxNumConversationRounds
        success, problem_type = self._extract_problem_type(user_input)
        if not success:
            print("Failed to identify the problem type")
            return False, None
        totExecutor = self._get_tot_executor(problem_type, self.prompt_type)
        if totExecutor is None:
            print("Problem type not supported yet")
            return False, None
        success, solution = totExecutor.run(user_input, max_num_rounds)
        return success, solution

    def _extract_problem_type(self, user_input):
        messages = self._generate_problem_type_query(user_input)
        temperature = HyperParams.DefaultTemperature
        max_tokens = HyperParams.DefaultMaxTokens
        reply = self.llm_agent.get_reply(messages, temperature, max_tokens)
        success, json_obj = utils.extract_json_from_text_string(reply)
        if not success:
            return False, None
        if not (consts.KEY_PROBLEM_TYPE in json_obj):
            return False, None
        try:
            problem_type = ProblemType(json_obj[consts.KEY_PROBLEM_TYPE])
        except:
            return False, None
        return True, problem_type

    def _generate_problem_type_query(self, user_input):
        msg_tmpl = """The user is asking "{}". What type of problem the user wants to solve? Please give the answer in the following JSON format: {{ "{}": "<problem_type>" }} where <problem_type> can only be "sudoku", "3sat", or "others"."""
        msg_content = msg_tmpl.format(user_input, consts.KEY_PROBLEM_TYPE)
        role = "user"
        msgs = self.llm_agent.compose_messages([role], [msg_content])
        return msgs

    def _get_tot_executor(self, problem_type: ProblemType, prompt_type: PromptGenType):
        if problem_type == ProblemType.Sudoku:
            return TreeOfThoughtExecutorForSudoku(self.config, prompt_type)
        elif problem_type == ProblemType.ThreeSAT:
            return TreeOfThoughtExecutorForThreeSAT()
        else:
            return None


class TreeOfThoughtExecutorBase(object):

    def __init__(self, prompt_type: PromptGenType) -> None:
        self.conversation_history = ""
        self.state_manager_visit_count_map = {}
        self.prompt_type = prompt_type
        if self.prompt_type == PromptGenType.PolicyModelBased:
            print("load predictor start...")
            self.predictor = LSTMPredictor()
            print("load predictor done...")

    def run(self, user_input, max_num_rounds):
        messages = self.prompter.generate_initial_prompt(user_input)
        for i in range(max_num_rounds):
            temperature = self._get_temperature()
            max_tokens = self._get_max_tokens()
            reply = self.llm_agent.get_reply(messages, temperature, max_tokens)
            self._incr_state_visit_count()

            is_initial = i == 0

            self.conversation_history += "\nA: {}".format(reply)

            if self._should_repeat(reply):
                continue
            success, solution = self.parser.parse_llm_reply(
                reply, self.prompt_type, is_initial)
            if not success:
                print("Failed to extract solution from the reply, will retry")
                continue  # retry
            print("******** run_tot solution:", solution)
            if self.prompt_type == PromptGenType.PolicyModelBased and not is_initial:
                candidate_states = numpy_matrices_to_custom_strings(solution)
                predicted_values = self.predictor.predict(candidate_states)
                print("values from policy model:", predicted_values)
                best_solution = solution[0]
                best_val = predicted_values[0]
                for i in range(len(solution)):
                    val = predicted_values[i]
                    sol = solution[i]
                    if val > best_val:
                        best_solution = sol
                        best_val = val

                print("best value:", best_val, "best sol:", best_solution)
                solution = best_solution

            self.state_manager.update_state(solution)

            rollback_steps = self._get_rollback_steps()
            solution_found, solution, curr_state_is_valid, messages = self.prompter.generate_prompt(
                self.conversation_history, rollback_steps)  # FIXME
            if solution_found:
                print(messages)  # FIXME: better print out
                return True, solution

            if not curr_state_is_valid:
                self.state_manager.rollback(rollback_steps)  # backtracking

        print("Sorry, unable to solve the problem within {} rounds of conversations.".format(
            max_num_rounds))
        return False, None

    def _incr_state_visit_count(self):
        if self.state_manager.get_current_state() is None:
            return
        curr_state_key = json.dumps(
            self.state_manager.get_current_state().tolist())
        if not (curr_state_key in self.state_manager_visit_count_map):
            self.state_manager_visit_count_map[curr_state_key] = 0
        self.state_manager_visit_count_map[curr_state_key] += 1
        print("\nVISIT COUNT for {}: {}\n".format(curr_state_key,
              self.state_manager_visit_count_map[curr_state_key]))

    def _get_parent_state_visit_count(self):
        parent_state = self.state_manager.get_state(rollback_steps=1)
        if parent_state is None:
            return 0
        parent_state_key = json.dumps(parent_state.tolist())
        if not (parent_state_key in self.state_manager_visit_count_map):
            return 0
        else:
            return self.state_manager_visit_count_map[parent_state_key]


class TreeOfThoughtExecutorForSudoku(TreeOfThoughtExecutorBase):

    def __init__(self, config, prompt_type) -> None:
        super().__init__(prompt_type)
        self.state_manager = SudokuStateManager()
        self.llm_agent = LLMAgent(config)
        self.parser = LLMReplyParserForSudoku()
        self.prompter = SudokuPrompter(
            self.llm_agent,
            self.state_manager,
            config.chatbot_max_context_length,
            config.chatbot_include_chat_history_in_query,
            self.prompt_type
        )

    def _should_repeat(self, llm_reply):
        return ("{" not in llm_reply)  # FIXME: make this check more generic

    def _get_temperature(self):
        return HyperParams.DefaultTemperature

    def _get_max_tokens(self):
        return HyperParams.DefaultMaxTokens

    def _get_rollback_steps(self):
        max_rollback_steps = self.state_manager.max_rollback_steps()
        parent_state_visit_count = self._get_parent_state_visit_count()
        if parent_state_visit_count >= HyperParams.MaxStateVisitCount:
            rollback_steps = 2  # should backtrack and explore other possibilities
        else:
            rollback_steps = 1

        if rollback_steps > max_rollback_steps:
            rollback_steps = max_rollback_steps

        curr_state_key = json.dumps(
            self.state_manager.get_current_state().tolist())

        print("State History:")
        for state in self.state_manager.sudoku_matrix_history:
            print("        State:", json.dumps(state.tolist()))
        print("max_rollback_steps: {}".format(max_rollback_steps))
        print("parent_state_visit_count: {}".format(parent_state_visit_count))
        print("ROLLBACK STEPS: {}\n".format(rollback_steps))
        return rollback_steps


class TreeOfThoughtExecutorForThreeSAT(TreeOfThoughtExecutorBase):
    def __init__(self, config) -> None:
        super().__init__()
        self.state_manager = None  # FIXME
        self.llm_agent = None  # FIXME
        self.parser = None  # FIXME
        self.prompter = None

    def _should_repeat(self, llm_reply):
        return False

    def _get_temperature(self):
        return HyperParams.DefaultTemperature

    def _get_max_tokens(self):
        return HyperParams.DefaultMaxTokens

    def _get_rollback_steps(self):
        return 1
