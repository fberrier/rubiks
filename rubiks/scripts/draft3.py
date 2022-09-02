########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
########################################################################################################################


if '__main__' == __name__:
    """ Just create a logger to print some stuff in this script """
    logger = Loggable(name=__file__)
    sp = Puzzle.factory(puzzle_type='sliding_puzzle', n=3).apply_random_moves(2)
    solver = Solver.factory(solver_type='bfs', puzzle_type='sliding_puzzle', n=3, time_out=3600, limit=2)
    solution = solver.solve(sp)
    logger.log_info(solution)



########################################################################################################################

