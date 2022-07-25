####################################################################
from rubiks.solvers.solver import Solver
from rubiks.puzzle.puzzle import Puzzle
####################################################################
if '__main__' == __name__:
    puzzle_type = Puzzle.sliding_puzzle
    n=3
    m=3
    nb_shuffles=8
    limit=15
    time_out=60
    solver_type=Solver.dfs
    check_optimal=True
    log_solution=True
    action_type=Solver.do_solve
    Solver.factory(**globals()).action()
####################################################################
