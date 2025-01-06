import pytest
from state.state import State
from checker.checker import SudokuChecker


def test_is_valid_correct_size_and_no_duplicates():
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "2", "3"], ["2", "3", "1"], ["3", "1", "2"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert is_valid
    assert reason == ""


def test_is_valid_incorrect_size():
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "2"], ["2", "3"], ["3", "1"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "sudoku size is not correct" in reason


def test_is_valid_number_out_of_range():
    # Out-of-range number (0 and 4 in a 3x3 Sudoku)
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["0", "2", "3"], ["4", "3", "1"], ["3", "1", "2"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "Cell [0][0] is filled with 0, which is not in the range of 1 to 3 for 3x3 sudoku." in reason


def test_is_valid_duplicate_in_row():
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "1", "3"], ["2", "3", "1"], ["3", "1", "2"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "duplicate number 1 in row 1" in reason


def test_is_valid_duplicate_in_column():
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "2", "3"], ["1", "3", "1"], ["3", "1", "2"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "duplicate number 1 in column 1" in reason


def test_is_valid_invalid_non_star_empty_cell():
    prev_state = State([["*", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["1", "2", "3"], ["2", "X", "1"], ["3", "1", "2"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "should be filled with '*' rather than X" in reason


def test_is_valid_overwrite_non_empty_cell():
    prev_state = State([["1", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])
    current_state = State([["2", "*", "*"], ["*", "*", "*"], ["*", "*", "*"]])

    checker = SudokuChecker(3)
    is_valid, reason = checker.is_valid(prev_state, current_state)
    assert not is_valid
    assert "Cell [0][0] has been filled with 1 previously" in reason


def test_is_solved_with_empty_cells():
    current_state = State([["1", "2", "*"], ["2", "3", "1"], ["3", "1", "2"]])

    checker = SudokuChecker(3)
    assert not checker.is_solved(current_state)


def test_is_solved_correct_board():
    current_state = State([["1", "2", "3"], ["2", "3", "1"], ["3", "1", "2"]])

    checker = SudokuChecker(3)
    assert checker.is_solved(current_state)
