from dotenv import load_dotenv
import os
from typing import List
import random
from prompter.prompter import Prompter, AzureOpenAIPrompter
from checker.checker import Checker, SudokuChecker
from state.state import State, StateParser


# prompt templates
class PromptTemplate:
    NOT_VALID_SINGLE_LEAF = """Unfortunately there is an error in your current solution {}. The reason it is invalid: {} Let us try again starting from this Sudoku board: {}. 
            In the next solution you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation. 
            We do NOT expect you to solve the problem in a single shot. You can return intermediate solutions with unfilled cells marked by "*".
            Please return your solution strictly following valid JSON format in the following JSON schema: {{ "rows": [] }}"""

    NOT_VALID_MULTI_LEAF = """Unfortunately there is an error in your current solution {}. The reason it is invalid: {} Let us try again starting from this Sudoku board: {}. Please return {} possible solutions. 
            In the next solution you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation. 
            We do NOT expect you to solve the problem in a single shot. You can return intermediate solutions with unfilled cells marked by "*". 
            Please return your solution strictly following valid JSON format in the following JSON schema: {{ "solutions": [{{ "rows": [] }}] }}"""

    VALID_SINGLE_LEAF = """Please try to solve this Sudoku puzzle {}.
            In the next solutions you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation. 
            Please return your solution strictly following valid JSON format in the following JSON schema: {{ "rows": [] }}"""

    VALID_MULTI_LEAF = """Please try to solve this Sudoku puzzle {}, return {} possible solutions.
            In the next solutions you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation. 
            Please return your solutions strictly following valid JSON format in the following JSON schema: {{ "solutions": [{{ "rows": [] }}] }}"""

    MEMORY = """These are the previous solutions we've already tried: {}. Based on this information, {}"""


class TreeOfThoughtController:
    """
    For managing game state and sending prompt based on signals from checker
    """

    def __init__(self,
                 value_model: None,
                 prompter: Prompter,
                 checker: Checker,
                 enable_memory: bool,
                 leaf_count: int,
                 puzzle_size: int,
                 max_rounds: int
                 ):
        self.value_model = value_model
        self.prompter = prompter
        self.checker = checker
        self.enable_memory = enable_memory
        self.leaf_count = leaf_count
        self.puzzle_size = puzzle_size  # 3
        self.max_rounds = max_rounds
        self.rollback_steps = 1  # controls how many steps per rollback

        self.states: List[State] = []  # manage current states
        self.history: List[State] = []  # for memory

    # state management
    def get_latest_state(self) -> State | None:
        if len(self.states) == 0:
            return
        return self.states[-1]

    def add_state(self, state: State):
        self.states.append(state)

    def add_history(self, state: State):
        self.history.append(state)

    def rollback_state(self):
        """pop current state and increment prev state visit by 1"""

        if len(self.states) == 0:
            return
        print(f"start rollback, current depth: {len(self.states)}")
        for i in range(self.rollback_steps):
            if len(self.states) == 0:
                return
            if len(self.states) > 1:
                self.states[-2].increment_visit_count()
            self.states.pop()  # 3 stands for 3x3 sudoku
        print(f"done rollback, current depth: {len(self.states)}")

    # runner
    def run(self):
        round = 1
        unexpected_error_count = 0
        solved = False
        initial_prompt = f"please solve this {self.puzzle_size}x{self.puzzle_size} sudoku puzzle"
        prompter_response = self.prompter.prompt(initial_prompt)

        while round < self.max_rounds and not solved:
            if self.leaf_count == 1:
                state = StateParser.parse_single(prompter_response)
                if not state:
                    # should retry, do not add to round
                    unexpected_error_count += 1
                    continue
            else:
                states = StateParser.parse_multi(prompter_response)
                if not states:
                    # should retry, do not add to round
                    unexpected_error_count += 1
                    continue

                if self.value_model:
                    # pick the highest q value one
                    state = max(states, key=self.value_model.predict)  # TODO
                else:
                    # randomly pick one
                    state = random.choice(states)

            self.add_history(state)

            prev_state = self.get_latest_state()
            is_valid, not_valid_reason = self.checker.is_valid(
                prev_state, state)
            if not is_valid:
                # not valid, rollback
                self.rollback_state()
            else:
                # check solved
                solved = self.checker.is_solved(state)
                # valid, should push state to states
                self.add_state(state)
                if solved:
                    # TODO: print states for observer
                    break

            # not solved, keep going deeper
            if not is_valid:
                if self.leaf_count == 1:
                    prompt_template = PromptTemplate.NOT_VALID_SINGLE_LEAF
                    # prev state, reason, cur state (after pop)
                    prompt = prompt_template.format(
                        str(prev_state),
                        not_valid_reason,
                        str(self.get_latest_state())
                    )
                else:
                    prompt_template = PromptTemplate.NOT_VALID_SINGLE_LEAF
                    # prev state, reason, leaf count, cur state (after pop)
                    prompt = prompt_template.format(
                        str(prev_state),
                        not_valid_reason,
                        self.leaf_count,
                        str(self.get_latest_state())
                    )
            else:
                # valid
                if self.leaf_count == 1:
                    prompt_template = PromptTemplate.VALID_SINGLE_LEAF
                    # cur state, leaf count
                    prompt = prompt_template.format(
                        str(self.get_latest_state())
                    )
                else:
                    prompt_template = PromptTemplate.VALID_MULTI_LEAF
                    # cur state, leaf count
                    prompt = prompt_template.format(
                        str(self.get_latest_state()),
                        self.leaf_count
                    )

            # add memory
            if self.enable_memory:
                prev_states = [str(s) for s in self.history]
                prompt = PromptTemplate.MEMORY.format(prev_states, prompt)

            print("sending prompt:", prompt)
            prompter_response = self.prompter.prompt(prompt)

            round += 1

        if solved:
            print("ok")
        else:
            print(
                f"unable to solve in {self.max_rounds} rounds of conversation with LLM")

        print(f"unexpected error occurred {unexpected_error_count} times")


def main():
    # prompter
    model = os.getenv("OPENAI_MODEL")
    model_version = os.getenv("OPENAI_API_VERSION")
    temperature = os.getenv("TEMPERATURE")
    max_tokens = os.getenv("MAX_TOKENS")
    # how many states should each step generate
    leaf_count = os.getenv("LEAF_COUNT")
    prompter = AzureOpenAIPrompter(
        model, model_version, temperature, max_tokens, leaf_count)

    puzzle_size = os.getenv("PUZZLE_SIZE")

    # checker
    checker = SudokuChecker(puzzle_size)

    # memory module
    enable_memory = os.getenv("ENABLE_MEMORY")

    value_model = None
    max_rounds = os.getenv("MAX_ROUNDS")

    totController = TreeOfThoughtController(
        value_model=value_model,
        prompter=prompter,
        checker=checker,
        enable_memory=enable_memory,
        leaf_count=leaf_count,
        puzzle_size=puzzle_size,
        max_rounds=max_rounds
    )

    totController.run()


if __name__ == "__main__":
    try:
        # Load the .env file
        load_dotenv(".env")
        print("Loaded .env file.")
    except Exception as e:
        print(f"Error loading .env file: {e}")
        exit(1)

    main()
