from dotenv import load_dotenv
import os
import random
import argparse
import json
import time

from prompter.prompter import Prompter, AzureOpenAIPrompter
from checker.checker import Checker, SudokuChecker
from state.state import State, StateParser
from observer.observer import Observer, LoggingObserver
from typing import List, Union


# prompt templates
class PromptTemplate:
    INITIAL_SINGLE_LEAF = """Please solve this {}x{} sudoku puzzle: {}, where * represents a cell to be filled in.
            In the next solution you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation.
            We do NOT expect you to solve the problem in a single shot. You can return intermediate solutions with unfilled cells marked by "*".
            Please return your solution strictly following valid JSON format in the following JSON schema: {{ "rows": [] }}.  Do not write an introduction, summary, or explanations."""

    INITIAL_MULTI_LEAF = """Please solve this {}x{} sudoku puzzle: {}, where * represents a cell to be filled in. Please return {} possible solutions.
            In the next solutions you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation.
            We do NOT expect you to solve the problem in a single shot. You can return intermediate solutions with unfilled cells marked by "*".
            Please return your solution strictly following valid JSON format in the following JSON schema: {{ "solutions": [{{ "rows": [] }}] }}.  Do not write an introduction, summary, or explanations."""

    NOT_VALID_SINGLE_LEAF = """Unfortunately there is an error in your current solution {}. The reason it is invalid: {} Let us try again starting from this Sudoku board: {}.
            In the next solution you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation.
            We do NOT expect you to solve the problem in a single shot. You can return intermediate solutions with unfilled cells marked by "*".
            Please return your solution strictly following valid JSON format in the following JSON schema: {{ "rows": [] }}.  Do not write an introduction, summary, or explanations."""

    NOT_VALID_MULTI_LEAF = """Unfortunately there is an error in your current solution {}. The reason it is invalid: {} Let us try again starting from this Sudoku board: {}. Please return {} possible solutions.
            In the next solution you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation.
            We do NOT expect you to solve the problem in a single shot. You can return intermediate solutions with unfilled cells marked by "*".
            Please return your solution strictly following valid JSON format in the following JSON schema: {{ "solutions": [{{ "rows": [] }}] }}.  Do not write an introduction, summary, or explanations."""

    VALID_SINGLE_LEAF = """Please try to solve this Sudoku puzzle {}.
            In the next solutions you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation.
            Please return your solution strictly following valid JSON format in the following JSON schema: {{ "rows": [] }}.  Do not write an introduction, summary, or explanations."""

    VALID_MULTI_LEAF = """Please try to solve this Sudoku puzzle {}, return {} possible solutions.
            In the next solutions you return, please just fill in a few cells since we will work together to solve the puzzle in multiple rounds of conversation.
            Please return your solutions strictly following valid JSON format in the following JSON schema: {{ "solutions": [{{ "rows": [] }}] }}.  Do not write an introduction, summary, or explanations."""

    MEMORY = """{} Also, These are the previous solutions we've already tried: {}. Please solve based on the information."""


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

        # observers
        self.observers: List[Observer] = []

    def add_observer(self, observer: Observer):
        self.observers.append(observer)
        self.prompter.add_observer(observer)
        self.checker.add_observer(observer)

    def remove_observer(self, observer: Observer):
        self.observers.remove(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.notify(message)

    # state management
    def get_latest_state(self) -> Union[State, None]:
        if len(self.states) == 0:
            return
        return self.states[-1]

    def add_state(self, state: State):
        self.states.append(state)

    # memory
    def add_history(self, state: State):
        self.history.append(state)

    def cleanup(self):
        """cleanup states before new puzzle"""
        self.states = []
        self.history = []

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
            self.states.pop()
        print(f"done rollback, current depth: {len(self.states)}")

    # runner: given puzzles in json, run ToT
    def run(self, puzzles):
        self.notify_observers("====== start controller run ======")
        self.notify_observers(f"leaf count: {self.leaf_count}")
        total_puzzles = len(puzzles)
        solved_puzzles = 0
        times = []  # list to store the time taken for each sudoku if solved

        for puzzle in puzzles:
            self.notify_observers(f"=== puzzle: {puzzle} ===\n")
            self.cleanup()
            round = 1
            unexpected_error_count = 0
            solved = False
            prompt = ""
            if self.leaf_count == 1:
                prompt = PromptTemplate.INITIAL_SINGLE_LEAF.format(
                    self.puzzle_size,
                    self.puzzle_size,
                    puzzle
                )
            else:
                prompt = PromptTemplate.INITIAL_MULTI_LEAF.format(
                    self.puzzle_size,
                    self.puzzle_size,
                    self.leaf_count,
                    puzzle
                )

            start_time = time.time()
            prompter_response = self.prompter.prompt(prompt)
            solution = None
            while round < self.max_rounds and not solved and unexpected_error_count < 5:
                if self.leaf_count == 1:
                    state = StateParser.parse_single(prompter_response)
                    if not state:
                        # should retry
                        unexpected_error_count += 1
                        prompter_response = self.prompter.prompt(prompt)
                        continue
                else:
                    states = StateParser.parse_multi(prompter_response)
                    if not states:
                        # should retry
                        unexpected_error_count += 1
                        prompter_response = self.prompter.prompt(prompt)
                        continue

                    if self.value_model:
                        # pick the highest q value one
                        state = max(
                            states, key=self.value_model.predict)  # TODO
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
                        solution = str(state)
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
                    prompt = PromptTemplate.MEMORY.format(prompt, prev_states)

                self.notify_observers(f"sending prompt: {prompt}")
                prompter_response = self.prompter.prompt(prompt)

                round += 1

            # out of while loop
            end_time = time.time()

            self.notify_observers(f"=== puzzle: {puzzle} ===")
            if solved:
                solved_time = end_time - start_time
                self.notify_observers(
                    f"solved {puzzle} in ** {round} ** rounds, solution: {solution}")
                self.notify_observers(
                    "solved in {:.4f} seconds".format(solved_time))
                times.append(end_time - start_time)
                solved_puzzles += 1
            elif unexpected_error_count >= 5:
                self.notify_observers(
                    f"something went wrong... unexpected error occurred {unexpected_error_count} times")
            else:
                self.notify_observers(
                    f"unable to solve {puzzle} in {round} rounds of conversation with LLM")

            self.notify_observers(
                f"unexpected error occurred {unexpected_error_count} times")

        self.notify_observers("\n=== SUMMARY ===")
        self.notify_observers(
            f"solved {solved_puzzles} puzzles out of {total_puzzles} puzzles")

        if len(times) > 0:
            average_time = sum(times) / len(times)
            self.notify_observers("Average solving time: {:.4f} seconds".format(
                average_time))
        self.notify_observers(
            f"Total LLM calls: {self.prompter.get_prompt_count()}")
        self.notify_observers(
            f"Total input tokens: {self.prompter.get_input_token_count()}")


def main():
    model = os.getenv("OPENAI_MODEL")
    model_version = os.getenv("OPENAI_API_VERSION")
    temperature = int(os.getenv("TEMPERATURE"))
    max_tokens = int(os.getenv("MAX_TOKENS"))
    max_rounds = int(os.getenv("MAX_ROUNDS"))
    # how many states should each step generate
    leaf_count = int(os.getenv("LEAF_COUNT"))

    # memory module
    enable_memory = bool(os.getenv("ENABLE_MEMORY"))

    # parse arguments
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-data", required=True, type=str, help="data path")
    parser.add_argument("-puzzle_size", required=True, type=int,
                        help="3 for 3x3, 4 for 4x4, must match given data")
    parser.add_argument("-type", required=True,
                        choices=["rule", "value"])
    parser.add_argument("-log", type=str,
                        help="file path for logging file")

    args = parser.parse_args()

    puzzle_size = args.puzzle_size

    # checker
    checker = SudokuChecker(puzzle_size)

    # prompter
    prompter = AzureOpenAIPrompter(
        model, model_version, temperature, max_tokens, leaf_count)

    # value model
    value_model = None
    if args.type == "value":
        # TODO: add value model
        value_model = None

    totController = TreeOfThoughtController(
        value_model=value_model,
        prompter=prompter,
        checker=checker,
        enable_memory=enable_memory,
        leaf_count=leaf_count,
        puzzle_size=puzzle_size,
        max_rounds=max_rounds
    )

    totController.add_observer(LoggingObserver(log_file=args.log))

    with open(args.data) as f:
        puzzles = json.load(f)
        totController.run(puzzles)


if __name__ == "__main__":
    try:
        # Load the .env file
        load_dotenv(".env")
        print("Loaded .env file.")
    except Exception as e:
        print(f"Error loading .env file: {e}")
        exit(1)

    main()
