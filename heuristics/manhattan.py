########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from heuristics.heuristics import Heuristic
from puzzle.sliding import SlidingPuzzle
########################################################################################################################


class Manhattan(Heuristic):
    """ TBD """

    puzzle_type = SlidingPuzzle


    def __init__(self, n, m=None):
        super().__init__(n=n, m=m)
        # build map of where each value should be
        self.goal_map = {}
        goal = 1
        size = n*m
        for row in range(n):
            for col in range(m):
                self.goal_map[goal] = (n, m)
                goal = goal + 1
                goal %= size
        (self.n, self.m) = tuple(self.puzzle_dimension())

    def cost_to_go_from_puzzle_impl(self, puzzle):
        print(self.n, self.m)

    

########################################################################################################################
