from abc import ABC, abstractmethod
from typing import Tuple
from state.state import State


class Checker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def is_valid(self, state: State) -> Tuple[bool, str]:
        """should return bool and a reason why it's not valid"""
        pass

    @abstractmethod
    def is_solved(self, state: State) -> bool:
        pass


class SudokuChecker(Checker):
    def __init__(self, puzzle_size: int):
        super().__init__()
        self.puzzle_size = puzzle_size  # 3, 4, or 5
        self.d = f"{self.puzzle_size}x{self.puzzle_size}"

    def is_valid(self, prev_state: State, state: State) -> Tuple[bool, str]:
        prev_board = prev_state.get_board()
        board = state.get_board()
        # 1: check size
        if len(board) != self.puzzle_size:
            return False, f"sudoku size is not correct, should be a {self.d} sudoku"
        for row in board:
            if not isinstance(row, list) or len(row) != self.puzzle_size:
                return False, f"sudoku size is not correct, should be a {self.d} sudoku"

        # 2: check if number is 1-n by row and col and no duplicates. Do not check block duplicates.
        # 3: check if not number, should be *
        # 4: check cannot fill in non-empty cell
        row_sets = [set() for _ in range(self.puzzle_size)]
        col_sets = [set() for _ in range(self.puzzle_size)]
        for r in range(self.puzzle_size):
            for c in range(self.puzzle_size):
                cell = board[r][c]
                try:
                    num = int(cell)
                    if prev_board[r][c] != "*":
                        return False, f"Cell [{r}][{c}] has been filled with {prev_board[r][c]} previously. We cannot set it to a diffent number."
                    if num in row_sets[r]:
                        return False, f"There is duplicate number {num} in row {r + 1}."
                    row_sets[r].add(num)
                    if num in col_sets[c]:
                        return False, f"There is duplicate number {num} in column {c + 1}."
                    col_sets[c].add(num)
                except ValueError:
                    # not a number
                    if cell != "*":
                        return False, f"The empty cell should be filled with '*' rather than {cell}."
                    continue

        return True, ""

    def is_solved(self, state: State) -> bool:
        """Only call this after calling is_valid"""
        board = state.get_board()
        for r in range(self.puzzle_size):
            for c in range(self.puzzle_size):
                if board[r][c] == "*":
                    return False
        return True
