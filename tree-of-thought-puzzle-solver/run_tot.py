import sys
from common.config import Config
from common.enums import *
from tot.tot import TreeOfThought
import json
import argparse
import os
import time

#
# Example Sudoku problems:
# '[[*, 3, 1], [*, 2, 3], [3, *, 2]]'
# '[[1, *, *, 2], [*, 1, *, 4], [*, 2, *, *], [*, *, 4, *]]'
#

from dotenv import load_dotenv

load_dotenv(".env")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
print(os.listdir())


if __name__ == "__main__":
    # jprint(os.environ["LANGCHAIN_PROJECT"])
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-data", required=True, type=str, help="data path")
    parser.add_argument("-prompt_type", required=True,
                        choices=["rule", "policy"])

    # Parse arguments
    args = parser.parse_args()
    # print(args.data)
    # Use arguments
    # if args
    dimension = args.data.split("/")[-1][:3]
    prompt_type = PromptGenType.RuleBased if args.prompt_type == "rule" else PromptGenType.PolicyModelBased
    path_to_config_yaml = "./config.yaml"
    config = Config(path_to_config_yaml)

    tot = TreeOfThought(config, prompt_type)
    initial_prompt = f"please solve this {dimension} sudoku puzzle"
    try:
        with open(args.data) as f:
            sudokus = json.load(f)
            times = []  # list to store the time taken for each sudoku if solved
            for sudoku in sudokus:
                prompt = initial_prompt + " " + sudoku + " " + \
                    "where * represents a cell to be filled in."
                start_time = time.time()
                success, solution = tot.run(prompt)
                end_time = time.time()
                if success:
                    times.append(end_time - start_time)

                print("Success :", success)
                print("Solution:", solution)

            if times:
                average_time = sum(times) / len(times)
                print("Average solving time: {:.4f} seconds".format(
                    average_time))
            else:
                print("No puzzles were solved, so no average time to compute.")

    except FileNotFoundError:
        print(f"Error: The file '{args.data}' was not found.")
        sys.exit(1)  # Exit the program with a non-zero status code
    except json.JSONDecodeError:
        print(
            f"Error: Failed to parse JSON from the file '{args.data}'. Please check the file format.")
        sys.exit(1)
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")
        sys.exit(1)
