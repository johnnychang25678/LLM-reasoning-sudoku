import logging
import matplotlib.pyplot as plt

class MCTSTree:
    """Custom in-memory tree for MCTS visualization."""
    def __init__(self):
        self.nodes = {}  # Node ID -> {'parent': parent_id, 'children': [], 'q_value': float}
        self.root = None

    def add_node(self, node_id, parent_id=None, q_value=None):
        if node_id not in self.nodes:
            self.nodes[node_id] = {'parent': parent_id, 'children': [], 'q_value': q_value}
            if parent_id:
                self.nodes[parent_id]['children'].append(node_id)
            if self.root is None:
                self.root = node_id

    def update_q_value(self, node_id, q_value):
        if node_id in self.nodes:
            self.nodes[node_id]['q_value'] = q_value

    def get_tree_structure(self):
        return self.nodes


class MCTSVisualizerHandler(logging.Handler):
    """Custom handler to visualize MCTS logs."""
    def __init__(self, tree: MCTSTree = None):
        super().__init__()
        if tree is None:
            tree = MCTSTree()
        self.tree = tree
        self.fig, self.ax = plt.subplots()
        plt.ion()  # Turn on interactive mode

    def emit(self, record):
        """Process log messages and update the tree."""
        log_entry = self.format(record)
        if "Selection" in log_entry:
            self.handle_selection(log_entry)
        elif "Expansion" in log_entry:
            self.handle_expansion(log_entry)
        elif "Backprop" in log_entry:
            self.handle_backprop(log_entry)
        self.update_visualization()

    def handle_selection(self, log_entry):
        """Parse selection logs."""
        # Extract node ID from the log entry (customize based on your log format)
        node_id = log_entry.split("Node: ")[-1].strip()
        self.tree.add_node(node_id)

    def handle_expansion(self, log_entry):
        """Parse expansion logs."""
        try:
            # Extract current and next states from the log entry
            parts = log_entry.split("current state: ")[-1].split(" -> next state: ")
            current_state = parts[0].strip()
            next_state = parts[1].strip()
            # Add to the tree or process as needed
            parent_id = current_state  # You can define a way to uniquely identify states
            node_id = next_state       # Similarly, generate a unique ID for the new node
            self.tree.add_node(node_id, parent_id)
        except Exception as e:
            print(f"Error parsing expansion log entry: {log_entry}")
            raise e

    def handle_backprop(self, log_entry):
        """Parse backpropagation logs."""
        # Extract node ID and Q-value from the log entry
        node_id, q_value = log_entry.split("Node: ")[-1].split(", Q-value: ")
        self.tree.update_q_value(node_id.strip(), float(q_value.strip()))

    def update_visualization(self):
        """Redraw the tree."""
        self.ax.clear()
        self.draw_tree(self.tree.root, 0.5, 1.0, 0.5)
        plt.draw()
        plt.pause(0.1)

    def draw_tree(self, node_id, x, y, spacing):
        """Recursively draw the tree."""
        if not node_id or node_id not in self.tree.nodes:
            return
        node = self.tree.nodes[node_id]
        self.ax.text(x, y, node_id, ha="center", va="center", bbox=dict(boxstyle="circle", fc="lightblue"))
        children = node['children']
        if children:
            for i, child_id in enumerate(children):
                child_x = x - spacing + i * (2 * spacing / max(1, len(children) - 1))
                child_y = y - 0.1
                self.ax.plot([x, child_x], [y, child_y], "k-")
                self.draw_tree(child_id, child_x, child_y, spacing / 2)