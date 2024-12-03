import argparse
import json
import logging
from typing import List
from mcts.MctsController import MctsController
from mcts.Visitor import MCTSVisitor
from mcts.Node import NodeFactory
from mcts.LangchainPolicyModel.LangchainGptPolicyModel import LangchainGptSudokuPolicyModel
from mcts.ValueModel import ValueModel

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_initial_states(json_file: str) -> List[List[List[str]]]:
    with open(json_file, 'r') as file:
        data = json.load(file)
        # Assuming the JSON file contains a list of states
        return data

def run_mcts_on_states(initial_states: List[List[List[str]]], iterations: int, controller: MctsController, visitor: MCTSVisitor) -> List[dict]:
    all_results = []
    for idx, state in enumerate(initial_states, start=1):
        logging.info(f"\n=== Running MCTS for Initial State {idx} ===")
        root_node = controller.run(state, iterations)
        
        # Visit the tree and collect data
        visitor.visit(root_node)
        nodes_data = visitor.get_nodes()
        
        # Add metadata to each node
        for node in nodes_data:
            node['initial_state'] = json.dumps(state)
        
        all_results.extend(nodes_data)
        
        # Clear visitor's nodes for the next state
        visitor.nodes.clear()
        
    return all_results

def export_results_to_csv(results: List[dict], csv_file: str):
    if not results:
        logging.warning("No results to export.")
        return
    try:
        visitor = MCTSVisitor()
        visitor.export_to_csv(csv_file)
        logging.info(f"Results successfully exported to {csv_file}")
    except Exception as e:
        logging.error(f"Failed to export results to CSV: {e}")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Run MCTS on Sudoku puzzles and export results to CSV.")
    parser.add_argument(
        '--input',
        type=str,
        help="Path to JSON file containing initial Sudoku states. If not provided, a default state will be used."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='mcts_results.csv',
        help="Path to the output CSV file."
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help="Number of MCTS iterations to perform per initial state."
    )
    
    args = parser.parse_args()
    
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
    
    # Initialize policy and value models
    policy_model = LangchainGptSudokuPolicyModel()
    # If using other policy models, initialize them here
    # policy_model = LangchainOllamaSudokuPolicyModel(model_name="llama3.1:70b-instruct-q2_K")
    # policy_model = OllamaSudokuPolicyModel()
    
    value_model = ValueModel()  # Initialize with actual implementation if available
    
    # Initialize NodeFactory
    node_factory = NodeFactory(policy_model)
    
    # Initialize MCTS Controller
    controller = MctsController(
        policy_model=policy_model,
        value_model=value_model,
        exploration_weight=1.0,
        max_simulate_depth=50,
        node_factory=node_factory
    )
    
    # Initialize Visitor
    visitor = MCTSVisitor()
    
    # Run MCTS on all initial states
    results = run_mcts_on_states(initial_states, args.iterations, controller, visitor)
    
    # Export results to CSV
    export_results_to_csv(results, args.output)

if __name__ == "__main__":
    main()
