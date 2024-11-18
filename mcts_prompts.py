# prompt

def gen_actions_prompt(state):
    return \
        f"""
        Given the following Sudoku state:
        {state}
        where * represents a cell to be filled in.
        Suggest the next possible states to solve this Sudoku. You can suggest more than one states you see fit. Do not repeat the same state.
        Respond only with valid JSON following below schema:
        Schema:
        {{
        "type": "object",
        "properties": {{
            "states": {{
            "type": "array",
            "items": {{
                "type": "object",
                "properties": {{
                "rows": {{
                    "type": "array",
                    "items": {{
                    "type": "array",
                    "items": {{
                        "type": "string"
                    }}
                    }}
                }}
                }},
                "required": [
                "rows"
                ]
            }}
            }}
        }},
        "required": [
            "states"
        ]
        }}
        Do not write an introduction or summary.
        """


def gen_is_terminal_prompt(state):
    return \
        f"""
    Given the following Sudoku state: {state}
    where * represents a cell to be filled in.
    Check if this Sudoku is solved correctly. Respond only with valid JSON following the schema below.
    {{
    "type": "object",
    "properties": {{
        "solved": {{
        "type": "boolean"
        }}
    }},
    "required": [
        "solved"
    ]
    }}
    Do not write an introduction or summary.
    """
