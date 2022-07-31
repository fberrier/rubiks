####################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver, Solution
####################################################################
if '__main__' == __name__:
    puzzle_type = Puzzle.sliding_puzzle
    tiles=[[3, 8, 6], [4, 1, 5], [0, 7, 2]]
    solver_type=Solver.astar
    heuristic_type=Heuristic.manhattan
    plus=True
    action_type=Solver.do_solve
    print(Solver.factory(**globals()).action().to_str([Solution.puzzle,
                                                       Solution.cost,
                                                       Solution.expanded_nodes,
                                                       Solution.success,
                                                       Solution.solver_name,
                                                       Solution.run_time]))
####################################################################