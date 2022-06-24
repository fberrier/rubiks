########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.sliding import SlidingPuzzle
########################################################################################################################


class Manhattan(Heuristic):
    """ This heuristic if admissible. The reason is that each non (zero==empty) tile
    has to move by at least its manhattan distance to solve the puzzle, and each move is only moving one tile
    at each step
    """

    @classmethod
    def known_to_be_admissible(cls):
        return True

    def __init__(self, n, m=None, **kw_args):
        if m is None:
            m = n
        kw_args.pop(__class__.puzzle_type, None)
        super().__init__(SlidingPuzzle, n=n, m=m, **kw_args)
        # build map of where each value should be
        self.goal_map = {}
        goal = 1
        size = n*m
        for row in range(n):
            for col in range(m):
                self.goal_map[goal] = (row, col)
                goal = goal + 1
                goal %= size
        (self.n, self.m) = tuple(self.get_puzzle_dimension())

    def cost_to_go_from_puzzle_impl(self, puzzle):
        puzzle = puzzle.clone()
        cost_to_go = 0
        row = 0
        col = 0
        while row < self.n:
            val = puzzle.tiles[row][col].item()
            goal = self.goal_map[val]
            cost = abs(row - goal[0]) + abs(col - goal[1])
            if val != 0:
                cost_to_go += cost
            col += 1
            if col == self.m:
                col = 0
                row += 1
        return cost_to_go

########################################################################################################################

