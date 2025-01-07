import json
from typing import List
import re


class State:
    """sudoku state"""

    def __init__(self, sudoku_board: List[List[str]]):
        self.sudoku = sudoku_board  # [["1", "2", "*"], []...]
        self.visit_count = 1  # when created, visit count = 1

    def increment_visit_count(self):
        self.visit_count += 1

    def get_board(self):
        return self.sudoku

    def __str__(self) -> str:
        return str(self.sudoku)


class StateParser:
    """methods to parse string from llm to State"""

    @staticmethod
    def parse_single(llm_response: str) -> State:
        # parse the llm response to state
        # return None if fail to parse

        # string to json
        sudoku_board_json_obj = StateParser._extract_json_from_text_string(
            llm_response)
        if sudoku_board_json_obj is None:
            return None

        key = "rows"
        if not (key in sudoku_board_json_obj):
            return None

        rows = sudoku_board_json_obj[key]

        # rectify the cells
        rectified_rows = []
        for row in rows:
            rectified_row = []
            for cell in row:
                rectified_cell = None
                if cell is None or str(cell).lower() == "none" or str(cell).lower() == "null":
                    rectified_cell = "*"
                else:
                    rectified_cell = str(cell)
                rectified_row.append(rectified_cell)
            rectified_rows.append(rectified_row)

        try:
            solution = rectified_rows
        except:
            print("parser error", "rows:", rectified_rows)
            return None
        state = State(solution)
        return state

    @staticmethod
    def parse_multi(llm_response: str) -> List[State]:
        # parse the llm response to state
        # return None if fail to parse

        # string to json
        sudoku_board_json_obj = StateParser._extract_json_from_text_string(
            llm_response)
        if sudoku_board_json_obj is None:
            return None

        key = "solutions"
        if not (key in sudoku_board_json_obj):
            return None

        sols = sudoku_board_json_obj[key]
        valid_sols = []
        key2 = "rows"
        for sol in sols:
            # rectify the cells
            if key2 not in sol:
                continue
            rows = sol[key2]
            rectified_rows = []
            for row in rows:
                rectified_row = []
                for cell in row:
                    rectified_cell = None
                    if cell is None or str(cell).lower() == "none" or str(cell).lower() == "null":
                        rectified_cell = "*"
                    else:
                        rectified_cell = str(cell)
                    rectified_row.append(rectified_cell)
                rectified_rows.append(rectified_row)

            try:
                valid_sol = rectified_rows
                valid_sols.append(valid_sol)
            except:
                print("parser error", "rows:", rectified_rows,
                      "valid_sols:", valid_sols)
                continue

        if len(valid_sols) == 0:
            print("no valid solutions parsed")
            return None

        return valid_sols

    @staticmethod
    def _extract_json_from_text_string(text_str: str):
        """extract json from string, return None if error"""
        try:
            lp_idx = text_str.index('{')
            rp_idx = text_str.rindex('}')
            json_str = text_str[lp_idx:rp_idx+1]
            json_str = json_str.replace("'", '"')
            # Quote unquoted * characters
            json_str = re.sub(
                r'(?<=\[|\s|,)(\*)(?=\s|,|\])', r'"\1"', json_str)
            # Quote unquoted integers
            json_str = re.sub(
                r'(?<=\[|\s|,)(\d+)(?=\s|,|\])', r'"\1"', json_str)

            json_obj = json.loads(json_str)
            return json_obj
        except Exception as e:
            print("extract_json_from_text_string error", e)
            return None
