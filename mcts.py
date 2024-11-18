import numpy as np
import random
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from mcts_prompts import gen_actions_prompt, gen_is_terminal_prompt


class Node:
    def __init__(self, state, parent=None, action=None, untried_actions=None):
        # state representation: [["2", "4", "1", "*"],["*", "1", "3", "2"],["3", "1", "4", "2"],["1", "3", "2", "4"]]
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.q_value = 0.0  # Estimated reward
        self.untried_actions = untried_actions or []

    def is_fully_expanded(self):
        # if all actions are tried, the node is fully expanded
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
        return f"Action: {self.action}, State: {self.state}, Q-value: {self.q_value}"

    def __repr__(self):
        return self.__str__()


class PolicyModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)

    def _check_state(self, cur_state, sudoku):
        """check if sudoku size is same as cur_state and if sudoku cells only consists of * or int"""
        if not isinstance(sudoku, list):
            return False
        rows, cols = len(sudoku), len(sudoku[0])
        if rows != len(cur_state) or cols != len(cur_state[0]):
            return False
        for i in range(rows):
            for j in range(cols):
                ele = sudoku[i][j]
                if not isinstance(ele, str):
                    return False
                valid = (ele.isdigit() and int(ele) > 0) or ele == "*"
                if not valid:
                    return False
        return True

    def generate_actions_and_states(self, cur_state):
        """Generate possible actions for a given Sudoku state by querying LLM."""
        # state:  "[[1, *, *], [*, 1, *], [*, 2, *]]"
        prompt = gen_actions_prompt(cur_state)
        # TODO: not sure if this is the correct way to inference
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        action_states = []
        # response should be a json
        try:
            next_states = json.loads(response)
        except json.JSONDecodeError:
            print("parse json error:", response)
            return []
        for next_state in next_states:
            if "rows" in next_state:
                sudoku = next_state["rows"]
                if self._check_state(cur_state, sudoku):
                    action = self._derive_action(cur_state, sudoku)
                    action_states.append((action, sudoku))
        # [("action...", "new_state")]
        return action_states

    def _derive_action(self, cur_state, next_state):
        actions = []
        for i in range(len(cur_state)):
            for j in range(len(cur_state[0])):
                before_value = cur_state[i][j]
                after_value = next_state[i][j]
                if before_value != after_value:
                    # Record the change
                    if before_value == "*":
                        action_str = f"Filled cell ({i+1},{j+1}) with {after_value}"
                    elif after_value == "*":
                        action_str = f"Emptied cell ({i+1},{j+1}) which was {before_value}"
                    else:
                        action_str = f"Changed cell ({i+1},{j+1}) from {before_value} to {after_value}"
                    actions.append(action_str)
        # Combine all actions into a single description
        if actions:
            action_description = "; ".join(actions) + "."
        else:
            action_description = "No changes made."

        out = f"Previous state: {cur_state}. Action taken: {action_description}"
        return out

    def check_is_terminal(self, state):
        prompt = gen_is_terminal_prompt(state)
        # TODO: not sure if this is the correct way to inference
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # response should be a json
        try:
            is_solved = json.loads(response)
        except json.JSONDecodeError:
            print("parse json error:", response)
            return False

        if "solved" in is_solved:
            return is_solved["solved"]
        return False


class ValueModel:
    def __init__(self):
        pass

    def generate_value(self, action):
        # given action, return q value
        pass


class MCTS:
    def __init__(self, policy_model: PolicyModel, value_model: ValueModel, exploration_weight=1.0, max_simulate_rounds=50):
        self.policy_model = policy_model  # Pretrained LLM for policy
        self.value_model = value_model  # Value model predicting Q-values
        self.exploration_weight = exploration_weight
        self.max_simulate_rounds = max_simulate_rounds

    def run(self, initial_state, num_simulations, dest=None):
        root = Node(initial_state)
        for _ in range(num_simulations):
            self._run_simulation(root)
        if dest:
            with open(dest, "w") as file:
                # Write the string representation of the Node
                file.write(str(root))
        else:
            print(root)

    def _run_simulation(self, root: Node):
        """Perform one simulation of MCTS."""
        # Step 1: Selection
        current_node = root
        while not self.is_terminal(current_node.state) and current_node.is_fully_expanded():
            current_node = current_node.best_child(self.exploration_weight)
        # Step 2: Expansion
        if not self.is_terminal(current_node.state):
            if not current_node.untried_actions:
                current_node.untried_actions = self.generate_actions_and_states(
                    current_node.state)
            if current_node.untried_actions:
                action, next_state = current_node.untried_actions.pop()  # randomly select one
                new_node = Node(state=next_state,
                                parent=current_node, action=action)
                current_node.children.append(new_node)
                current_node = new_node

        # Step 3: Simulation
        reward = self.simulate(current_node.state)

        # Step 4: Backpropagation
        self.backpropagate(current_node, reward)

    def is_terminal(self, state):
        """Check if the given state is a terminal state."""
        return self.policy_model.check_is_terminal(state)

    def generate_actions_and_states(self, state):
        """Generate a list of possible actions using the policy model."""
        return self.policy_model.generate_actions_and_states(
            state)

    def simulate(self, state):
        """Simulate the game to the end from the given state and return the reward."""
        # Use a rollout policy to complete the puzzle
        rounds = 1
        while not self.is_terminal(state) and rounds < self.max_simulate_rounds:
            actions_states = self.policy_model.generate_actions_and_states(
                state)
            if not actions_states:
                # if no valid actions, the path doesn't work
                return -1
            if not self.value_model:
                # randomly select one
                action, next_state = random.choice(actions_states)
            else:
                # select highest q_value
                action, next_state = max(
                    actions_states, key=lambda x: self.value_model.generate_value(x[0]))
            state = next_state
            rounds += 1
        return 1 if self.is_terminal(state) else -1

    def backpropagate(self, node: Node, reward):
        """Update Q-values and visit counts along the path back to the root."""
        while node is not None:
            node.visits += 1
            node.q_value += (reward - node.q_value) / node.visits
            node = node.parent


if __name__ == "__main__":
    policy_model = PolicyModel("meta-llama/Meta-Llama-3-8B")
    value_model = None
    mcts = MCTS(policy_model=policy_model, value_model=value_model)
    sudoku = [["1", "*", "*"], ["*", "1", "*"], ["*", "2", "*"]]
    mcts.run(initial_state=sudoku, num_simulations=10)


"""
{
  "states": [
    {
      "rows": [
        ["2", "4", "1", "3"],
        ["4", "1", "3", "2"],
        ["3", "1", "4", "2"],
        ["1", "3", "2", "4"]
      ]
    },
    {
      "rows": [
        ["3", "2", "1", "4"],
        ["4", "1", "2", "3"],
        ["1", "4", "3", "2"],
        ["2", "3", "4", "1"]
      ]
    }
  ]
}
"""
