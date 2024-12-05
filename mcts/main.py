import argparse
import json
import logging
import os
import csv
import random
from typing import List
from mcts.MctsController import MctsController
from mcts.MctsObserver.ExportTreeObserver import ExportTreeObserver
from mcts.Visitor import MCTSVisitor
from mcts.Node import NodeFactory
from mcts.LangchainPolicyModel.LangchainGptPolicyModel import LangchainGptSudokuPolicyModel
from mcts.ValueModel import ValueModel

def setup_logging(log_file: str):
    """
    Configures logging to output to both console and a specified log file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create StreamHandler for console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Create FileHandler for log file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def load_initial_states(json_file: str) -> List[List[List[str]]]:
    with open(json_file, 'r') as file:
        data = json.load(file)
        # Assuming the JSON file contains a list of states
        return data

def export_results_to_csv(results: List[dict], csv_file: str):
    if not results:
        logging.warning("No results to export.")
        return
    try:
        with open(csv_file, mode='w', newline='') as file:
            fieldnames = ['state', 'q_value', 'initial_state']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logging.info(f"Results successfully exported to {csv_file}")
    except Exception as e:
        logging.error(f"Failed to export results to CSV: {e}")

def run_mcts_on_puzzle(state: List[List[str]], iterations: int, output_dir: str, puzzle_idx: int):
    """
    Initializes and runs MCTS for a single puzzle, exports iteration trees.
    """
    logging.info(f"\n=== Running MCTS for Puzzle {puzzle_idx} ===")

    # Initialize Visitor
    visitor = MCTSVisitor()

    # Initialize Observers
    observers = [ExportTreeObserver(dirname=output_dir, prefix=f"mcts_puzzle_{puzzle_idx}")]

    # Initialize policy and value models
    policy_model = LangchainGptSudokuPolicyModel()
    # If using other policy models, initialize them here
    # policy_model = LangchainOllamaSudokuPolicyModel(model_name="llama3.1:70b-instruct-q2_K")
    # policy_model = OllamaSudokuPolicyModel()

    value_model = None  # Initialize with actual implementation if available

    # Initialize NodeFactory
    node_factory = NodeFactory(policy_model)

    # Initialize MCTS Controller with observers
    controller = MctsController(
        policy_model=policy_model,
        value_model=value_model,
        exploration_weight=1.0,
        max_simulate_depth=50,
        node_factory=node_factory,
        observers=observers
    )

    # Run MCTS
    root_node = controller.run(state, iterations)

    # Visit the tree and collect data
    visitor.visit(root_node)
    nodes_data = visitor.get_nodes()

    # Add metadata to each node
    for node in nodes_data:
        node['initial_state'] = json.dumps(state)

    # Export collected nodes to a CSV specific to the puzzle
    export_results_to_csv(nodes_data, os.path.join(output_dir, f"mcts_puzzle_{puzzle_idx}_results.csv"))

def main():
    parser = argparse.ArgumentParser(description="Run MCTS on Sudoku puzzles and export results.")
    parser.add_argument(
        '--input',
        type=str,
        help="Path to JSON file containing initial Sudoku states. If not provided, a default state will be used."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='mcts_results',
        help="Path to the output directory."
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help="Number of MCTS iterations to perform per initial state."
    )
    parser.add_argument(
        '--shuffle',
        type=bool,
        default=False,
        help="Shuffle the initial states before running MCTS."
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Setup logging after output directory is created to save log file there
    log_file = os.path.join(args.output, 'mcts.log')
    setup_logging(log_file)
    logging.info("Logging initialized.")

    # Load initial states
    if args.input:
        try:
            initial_states = load_initial_states(args.input)
            logging.info(f"Loaded {len(initial_states)} initial states from {args.input}")
        except Exception as e:
            logging.error(f"Failed to load initial states from {args.input}: {e}")
            return
    else:
        # Define a default initial state if no input file is provided
        initial_states = [
            [['1', '*', '*'],
             ['*', '1', '*'],
             ['*', '2', '*']]
        ]
        logging.info("No input file provided. Using default initial state.")

    if args.shuffle:
        random.shuffle(initial_states)
        logging.info("Shuffled initial states.")

    # Iterate over each puzzle and run MCTS individually
    for idx, state in enumerate(initial_states, start=1):
        run_mcts_on_puzzle(state, args.iterations, args.output, idx)

if __name__ == "__main__":
    main()
