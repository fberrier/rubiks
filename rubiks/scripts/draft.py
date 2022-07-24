""" imports >>>>>> """
from math import inf
from rubiks . puzzle . puzzle import Puzzle
from rubiks . solvers . solver import Solver
""" imports <<<<<< """
action_type = Solver . do_solve
n = 2
puzzle_type = Puzzle . sliding_puzzle
solver_type = Solver . naive
nb_shuffles = inf
print ( Solver . factory ( ** globals () ) . action () )