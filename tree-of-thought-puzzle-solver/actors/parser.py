import numpy as np
import common.utils as utils
import common.consts as consts
from common.enums import PromptGenType


class LLMReplyParserBase(object):

    def __init__(self) -> None:
        pass

    def parse_llm_reply(self, llm_reply):
        pass


class LLMReplyParserForSudoku(LLMReplyParserBase):

    def __init__(self) -> None:
        pass

    def parse_llm_reply(self, llm_reply, prompt_type: PromptGenType, is_initial: bool):
        # parse json and convert to sudoku board
        success, json_obj = utils.extract_json_from_text_string(llm_reply)
        if not success:
            print("None 1")
            return False, None
        if prompt_type == PromptGenType.PolicyModelBased and not is_initial:
            # will have multiple solutions
            parse_result = self.extract_sudoku_board_from_solutions(json_obj)
        else:
            parse_result = self.extract_sudoku_board_from_rows(json_obj)
        return parse_result

    def extract_sudoku_board_from_rows(self, sudoku_board_json_obj):
        if not (consts.KEY_ROWS in sudoku_board_json_obj):
            return False, None

        rows = sudoku_board_json_obj[consts.KEY_ROWS]

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
            solution = np.matrix(rectified_rows)
        except:
            print("parser error", "rows:", rectified_rows)
            return False, None

        return True, solution

    def extract_sudoku_board_from_solutions(self, sudoku_board_json_obj):
        """extract sudokus from json:
        {
            "solutions": [
                {
                    "rows": [[], [], []],
                },
                {
                    "rows": [[], [], []],
                },
                {
                    "rows": [[], [], []],
                }
            ]

        }
        """
        print("1 extract_sudoku_board_from_solutions", sudoku_board_json_obj)
        if not (consts.KEY_SOLUTIONS in sudoku_board_json_obj):
            print("None 2")
            return False, None
        # if not (consts.KEY_ROWS in sudoku_board_json_obj):
        #     return False, None
        # rows = sudoku_board_json_obj[consts.KEY_ROWS]
        sols = sudoku_board_json_obj[consts.KEY_SOLUTIONS]
        print("2 extract_sudoku_board_from_solutions", sols)
        valid_sols = []
        for sol in sols:
            # rectify the cells
            if consts.KEY_ROWS not in sol:
                continue
            rows = sol[consts.KEY_ROWS]
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
                valid_sol = np.matrix(rectified_rows)
                valid_sols.append(valid_sol)
            except:
                print("parser error", "rows:", rectified_rows,
                      "valid_sols:", valid_sols)
                continue

        print("3 extract_sudoku_board_from_solutions", valid_sols)
        if len(valid_sols) == 0:
            print("None 3")
            return False, None
        return True, valid_sols
