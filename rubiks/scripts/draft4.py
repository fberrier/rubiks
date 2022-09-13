####################################################################
from rubiks.solvers.solver import Solver
from rubiks.puzzle.puzzle import Puzzle
####################################################################
if '__main__' == __name__:
    puzzle_type = Puzzle.sliding_puzzle
    n=3
    m=3
    nb_shuffles=5
    solver_type=Solver.bfs
    check_optimal=True
    action_type=Solver.do_solve
    print(Solver.factory(**globals()).action())
####################################################################