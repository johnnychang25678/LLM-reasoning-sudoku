import numpy as np

from mcts.OllamaPolicyModel.OllamaPolicyModel import PolicyModel


class Node:
    def __init__(self, state, parent=None, from_action=None, untried_actions=None, is_terminal=False):
        # state representation: [["2", "4", "1", "*"],["*", "1", "3", "2"],["3", "1", "4", "2"],["1", "3", "2", "4"]]
        self.state = state
        self.from_action = from_action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.q_value = 0.0  # Estimated reward
        self.is_terminal = is_terminal
        self.untried_actions = untried_actions

    def is_fully_expanded(self):
        # if all actions are tried, the node is fully expanded
        if not self.untried_actions:
            return True
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight):
        """Select the child with the highest UCT value."""
        max_uct = float("-inf")
        max_child = None
        for child in self.children:
            uct = child.q_value + exploration_weight * \
                np.sqrt(np.log(self.visits + 1) / (child.visits + 1))
            if uct > max_uct:
                max_uct = uct
                max_child = child
        return max_child

    def __str__(self):
        return f"---\nAction: {self.from_action}, State: {self.state}, Q-value: {self.q_value}\n---\n"

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def make_node(state, parent=None, from_action=None, untried_actions=None, policy_model=None):
        node = Node(state, parent, from_action, untried_actions)
        if not policy_model:
            return node
        
        # check if the state is terminal
        terminal, reward = policy_model.check_is_terminal_and_give_reward(state)
        if terminal:
            node.q_value = reward
            node.is_terminal = True
            return node
        
        # generate actions and states
        actions_states = policy_model.generate_actions_and_states(state)
        if not actions_states:
            return node
        
        node.untried_actions = actions_states
        return node
    
class NodeFactory:
    def __init__(self, policy_model: PolicyModel):
        self.policy_model = policy_model
    
    def make_node(self, state, parent=None, from_action=None, untried_actions=None):
        return Node.make_node(state, parent, from_action, untried_actions, self.policy_model)