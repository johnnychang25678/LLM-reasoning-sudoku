from mcts.Node import Node

class BaseMctsObserver:
    def __init__(self):
        pass

    def observe_iteration_end(self, iteration: int, root_node: Node):
        pass
