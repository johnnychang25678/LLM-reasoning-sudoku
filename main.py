from got.got import GroupOfThoughts
from got.got import ToTSolver
from got.config.config_loader import Config
from got.game import SudokuGame

if __name__ == "__main__":
    config = Config()
    agents = config.get_agents()
    print(agents)
    game = SudokuGame()
    tot_solver = ToTSolver(agents)
    got = GroupOfThoughts(game=SudokuGame(), strategy=tot_solver)
    got.run()
