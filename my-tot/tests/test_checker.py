import pytest
from state.state import State
from checker.checker import SudokuChecker


def test_is_valid_correct_size_and_no_duplicates():
    # Valid 3x3 board
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert is_valid
    assert reason == ""


def test_is_valid_incorrect_size():
    # Board with incorrect size (3x2 instead of 3x3)
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "2"], ["4", "5"], ["7", "8"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "sudoku size is not correct" in reason


def test_is_valid_duplicate_in_row():
    # Duplicate in the first row
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "1", "3"], ["4", "5", "6"], ["7", "8", "9"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "duplicate number 1 in row 1" in reason


def test_is_valid_duplicate_in_column():
    # Duplicate in the first column
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "2", "3"], ["1", "5", "6"], ["7", "8", "9"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "duplicate number 1 in column 1" in reason


def test_is_valid_invalid_non_star_empty_cell():
    # Board contains invalid non-star placeholder
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "2", "3"], ["4", "X", "6"], ["7", "8", "9"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "should be filled with '*' rather than X" in reason


def test_is_valid_overwrite_non_empty_cell():
    # Overwriting a pre-filled cell
    prev_state = State([["1", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["2", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "Cell [0][0] has been filled with 1 previously" in reason


def test_is_solved_with_empty_cells():
    # Board is not solved (contains empty cells)
    current_state = State([["1", "2", "*"], ["4", "5", "6"], ["7", "8", "9"]])

    checker = SudokuChecker(3)
    assert not checker.is_solved(current_state)


def test_is_solved_correct_board():
    # Board is fully solved
    current_state = State([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])

    checker = SudokuChecker(3)
    assert checker.is_solved(current_state)
