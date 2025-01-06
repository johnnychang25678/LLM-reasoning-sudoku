import pytest
from state.state import State, StateParser


def test_state_initialization():
    # Test the initialization of the State class
    sudoku_board = [["1", "2", "*"], ["3", "4", "*"], ["*", "*", "5"]]
    state = State(sudoku_board)
    assert state.get_board() == sudoku_board
    assert state.visit_count == 1


def test_state_increment_visit_count():
    # Test the increment_visit_count method
    sudoku_board = [["1", "2", "*"], ["3", "4", "*"], ["*", "*", "5"]]
    state = State(sudoku_board)
    state.increment_visit_count()
    assert state.visit_count == 2


def test_state_str_representation():
    # Test the __str__ method
    sudoku_board = [["1", "2", "*"], ["3", "4", "*"], ["*", "*", "5"]]
    state = State(sudoku_board)
    assert str(state) == str(sudoku_board)


def test_parse_single_valid_response():
    # Test parse_single with a valid LLM response
    llm_response = "Some text... {'rows': [['1', '2', '*'], ['3', '4', '*'], ['*', '*', '5']]}"
    state = StateParser.parse_single(llm_response)
    assert state is not None
    assert state.get_board() == [["1", "2", "*"],
                                 ["3", "4", "*"], ["*", "*", "5"]]


def test_parse_single_invalid_response():
    # Test parse_single with an invalid LLM response
    llm_response = "Some text... {'invalid_key': [['1', '2', '*'], ['3', '4', '*'], ['*', '*', '5']]}"
    state = StateParser.parse_single(llm_response)
    assert state is None


def test_parse_multi_valid_response():
    # Test parse_multi with a valid LLM response
    llm_response = """Some text...
    {
        "solutions": [
            {"rows": [["1", "2", "*"], ["3", "4", "*"], ["*", "*", "5"]]},
            {"rows": [["5", "*", "4"], ["*", "1", "2"], ["3", "2", "1"]]}
        ]
    }"""
    states = StateParser.parse_multi(llm_response)
    assert states is not None
    assert len(states) == 2
    assert states[0] == [["1", "2", "*"], ["3", "4", "*"], ["*", "*", "5"]]
    assert states[1] == [["5", "*", "4"], ["*", "1", "2"], ["3", "2", "1"]]


def test_parse_multi_invalid_response():
    # Test parse_multi with an invalid LLM response
    llm_response = "Some text... {'invalid_key': [{'rows': [['1', '2', '*'], ['3', '4', '*'], ['*', '*', '5']]}]}"
    states = StateParser.parse_multi(llm_response)
    assert states is None


def test_parse_multi_no_solutions_key():
    # Test parse_multi with missing "solutions" key
    llm_response = """Some text... {'rows': [['1', '2', '*'], ['3', '4', '*'], ['*', '*', '5']]}"""
    states = StateParser.parse_multi(llm_response)
    assert states is None
