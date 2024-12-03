import random
from mcts.LangchainPolicyModel.LangchainGptPolicyModel import LangchainGptSudokuPolicyModel
from mcts.LangchainPolicyModel.LangchainOllamaPolicyModel import LangchainOllamaSudokuPolicyModel
from mcts.OllamaPolicyModel.OllamaPolicyModel import OllamaSudokuPolicyModel, PolicyModel
from mcts.ValueModel import ValueModel
from mcts.Node import Node, NodeFactory
from typing import List
import logging
logger = logging.getLogger(__name__)

class MctsController:
    def __init__(self, policy_model: PolicyModel, value_model: ValueModel, exploration_weight=1.0, max_simulate_depth=50, node_factory: NodeFactory = None):
        self.policy_model = policy_model
        self.value_model = value_model
        self.exploration_weight = exploration_weight
        self.max_simulate_depth = max_simulate_depth
        self.node_factory = node_factory

    def run(self, initial_state: List[List[str]], iterations: int):
        logger.info(f"Starting MCTS with {iterations} iterations")
        logger.info(f"Initial state: {initial_state}")
        root_node = self.node_factory.make_node(initial_state)
        return self._run(root_node, iterations)

    def _run(self, root_node: Node, iterations: int):
        for i in range(iterations):
            logger.info(f"\nStarting iteration {i+1}/{iterations}")
            self._run_iteration(root_node)

        logger.info(f"MCTS completed. Root node visits: {root_node.visits}")
        return root_node

    def _run_iteration(self, root_node: Node):
        node_to_expand = self._select_node_to_expand(root_node)
        new_node = self._expand_node(node_to_expand)
        reward = self._rollout_from_node(new_node)
        self._backpropagate(new_node, reward)

    def _select_node_to_expand(self, start_node: Node):
        current_node = start_node
        depth = 0
        while True:
            logger.info(f"Selection - Depth {depth}, Current node state: {current_node.state}")
            if current_node.is_terminal:
                logger.info("Selection - Found terminal node")
                return current_node
            elif current_node.is_fully_expanded():
                logger.info("Selection - Node fully expanded, selecting best child")
                current_node = self._select_best_child(current_node)
                depth += 1
            else:
                logger.info("Selection - Found node to expand")
                return current_node

    def _select_best_child(self, node: Node):
        best_child = node.best_child(self.exploration_weight)
        logger.info(f"Selected child with Q-value: {best_child.q_value:.3f}, Visits: {best_child.visits}")
        return best_child

    def _expand_node(self, node_to_expand: Node):
        if node_to_expand.is_terminal:
            logger.info("Expansion - Node is terminal, no expansion needed")
            return node_to_expand
        elif not node_to_expand.untried_actions:
            logger.info("Expansion - No untried actions available")
            return node_to_expand
        else:
            action, next_state = node_to_expand.untried_actions.pop()
            new_node = self.node_factory.make_node(next_state, node_to_expand, action)
            node_to_expand.children.append(new_node)
            logger.info(f"Expansion - Created new node with action {action}")
            logger.info(f"New state: {next_state}")
            return new_node

    def _rollout_from_node(self, node: Node):
        rounds = 0
        current_state = node.state
        logger.info(f"Starting rollout from state: {current_state}")
        
        while rounds < self.max_simulate_depth:
            terminal, reward = self.policy_model.check_is_terminal_and_give_reward(current_state)
            if terminal:
                logger.info(f"Rollout - Found terminal state with reward: {reward}")
                return reward
            
            actions_states = self.policy_model.generate_actions_and_states(current_state)
            if not actions_states:
                logger.info("Rollout - No valid actions available")
                return -1
            
            if self.value_model:
                _, next_state = max(actions_states, key=lambda x: self.value_model.generate_value(x[0]))
                logger.info("Rollout - Selected action using value model")
            else:
                _, next_state = random.choice(actions_states)
                logger.info("Rollout - Selected random action")

            current_state = next_state
            rounds += 1
            logger.info(f"Rollout - Round {rounds}, Current state: {current_state}")

        logger.info("Rollout - Reached maximum depth")
        return 0

    def _backpropagate(self, node: Node, reward: float):
        depth = 0
        logger.info(f"Starting backpropagation with reward: {reward}")
        while node is not None:
            old_value = node.q_value
            node.visits += 1
            node.q_value += (reward - node.q_value) / node.visits
            logger.info(f"Backprop - Depth {depth}, Visits: {node.visits}, Q-value: {old_value:.3f} -> {node.q_value:.3f}")
            node = node.parent
            depth += 1

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # policy_model = OllamaSudokuPolicyModel()
    # policy_model = LangchainOllamaSudokuPolicyModel(model_name="llama3.1:70b-instruct-q2_K")
    policy_model = LangchainGptSudokuPolicyModel()
    node_factory = NodeFactory(policy_model)
    controller = MctsController(policy_model=policy_model, value_model=None, exploration_weight=1.0, max_simulate_depth=50, node_factory=node_factory)
    print(controller.run([['1', '*', '*'], ['*', '1', '*'], ['*', '2', '*']], 100))
