import csv

class MCTSVisitor:
    def __init__(self):
        self.nodes = []

    def visit(self, node):
        self._traverse(node)

    def _traverse(self, node):
        self.nodes.append({
            'state': node.state,
            'q_value': node.q_value
        })
        for child in node.children:
            self._traverse(child)

    def get_nodes(self):
        return self.nodes

    def export_to_csv(self, file_path):
        with open(file_path, mode='w', newline='') as csv_file:
            fieldnames = ['state', 'q_value']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for node in self.nodes:
                writer.writerow(node)
