####################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver, Solution
####################################################################
if '__main__' == __name__:
    puzzle_type = Puzzle.sliding_puzzle
    tiles=[[3, 1, 2, 0, 5], [7, 6, 4, 8, 9]]
    solver_type=Solver.astar
    heuristic_type=Heuristic.manhattan
    time_out=60
    plus=False
    action_type=Solver.do_solve
    print(Solver.factory(**globals()).action().to_str([Solution.puzzle,
                                                       Solution.cost,
                                                       Solution.expanded_nodes,
                                                       Solution.success,
                                                       Solution.solver_name,
                                                       Solution.run_time]))
####################################################################