# prompt

def gen_actions_prompt(state):
    return \
        f"""
        Given the following Sudoku state:
        {state}
        where * represents a cell to be filled in.
        Suggest the next possible states to solve this Sudoku. You can suggest more than one states you see fit. Do not repeat the same state.
        The sudoku cell should be filled with either str(int) or "*".
        Respond only with valid JSON with below example:
        {{
            "states": [
                {{
                "rows": [
                    ["*", "*", "1"],
                    ["*", "2", "3"],
                    ["3", "*", "2"],
                ]
                }},
                {{
                "rows": [
                    ["2", "*", "1"],
                    ["1", "2", "3"],
                    ["3", "*", "2"],
                ]
                }}
            ]
        }} 
        Do not write an introduction or summary.
        """


def gen_is_terminal_prompt(state):
    return \
        f"""
    Given the following Sudoku state: {state}
    where * represents a cell to be filled in.
    Check if this Sudoku is solved correctly. Respond only with valid JSON with below example:
    {{
        "solved": false
    }}
    Do not write an introduction or summary.
    """
