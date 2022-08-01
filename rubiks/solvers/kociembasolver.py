########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from kociemba import solve as kociemba_solve
########################################################################################################################
from rubiks.puzzle.rubikscube import RubiksCube
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class KociembaSolver(Solver):
    """ Rubik's solver from Kociemba """

    @classmethod
    def know_to_be_optimal(cls):
        return False

    def solve_impl(self, puzzle, **kw_args) -> Solution:
        assert isinstance(puzzle, RubiksCube) and puzzle.dimension() == 3
        solution = kociemba_solve(puzzle.to_kociemba())


########################################################################################################################
