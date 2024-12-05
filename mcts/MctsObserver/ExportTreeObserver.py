from mcts.MctsObserver.BaseObserver import BaseMctsObserver
from mcts.Node import Node
from mcts.Visitor import MCTSVisitor


class ExportTreeObserver(BaseMctsObserver):
    def __init__(self, dirname: str, prefix: str):
        self.dirname = dirname
        self.prefix = prefix

    def observe_iteration_end(self, iteration: int, root_node: Node):
        visitor = MCTSVisitor()
        visitor.visit(root_node)
        # results = visitor.get_nodes()
        visitor.export_to_csv(f"{self.dirname}/{self.prefix}_iteration_{iteration}.csv")

        
