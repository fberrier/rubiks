########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf
########################################################################################################################
from rubiks.search.bfsstrategy import BreadthFirstSearch
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class QSolver(Solver):

    def known_to_be_optimal(self):
        """ if it does not time out, it is definitely optimal """
        return False

    def solve_impl(self, puzzle, **kw_args):
        # @todo Francois: solver based on policies .... can hook DQL here
        return Solution(inf,
                        [],
                        inf,
                        puzzle)

########################################################################################################################
