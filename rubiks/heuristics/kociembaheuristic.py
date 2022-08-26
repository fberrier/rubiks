########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.kociembasolver import KociembaSolver
########################################################################################################################


class KociembaHeuristic(Heuristic):
    """ Makes use of Kociemba solver """

    @classmethod
    def known_to_be_admissible(cls):
        return False

    def __init__(self, **kw_args):
        Heuristic.__init__(self, **kw_args)
        self.solver = {n: KociembaSolver(puzzle_type=Puzzle.rubiks_cube, n=n) for n in[2, 3]}

    def cost_to_go_from_puzzle_impl(self, puzzle):
        return self.solver[puzzle.n].solve(puzzle).cost

########################################################################################################################

