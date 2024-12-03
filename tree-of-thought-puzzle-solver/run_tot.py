import sys
from common.config import Config
from tot.tot import TreeOfThought
import json
import argparse

#
# Example Sudoku problems:
# '[[*, 3, 1], [*, 2, 3], [3, *, 2]]'
# '[[1, *, *, 2], [*, 1, *, 4], [*, 2, *, *], [*, *, 4, *]]'
#


if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-data", required=True, type=str, help="data path")

    # Parse arguments
    args = parser.parse_args()
    # print(args.data)
    # Use arguments
    # if args
    dimension = args.data.split("/")[-1][:3]
    path_to_config_yaml = "./config.yaml"
    config = Config(path_to_config_yaml)
    tot = TreeOfThought(config)
    initial_prompt = f"please solve this {dimension} sudoku puzzle"
    try:
        with open(args.data) as f:
            sudokus = json.load(f)
            for sudoku in sudokus:
                prompt = initial_prompt + " " + sudoku + " " + \
                    "where * represents a cell to be filled in."
                # print("ppp", prompt)
                success, solution = tot.run(prompt)
                print("")
                print("Success :", success)
                print("Solution:", solution)
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
