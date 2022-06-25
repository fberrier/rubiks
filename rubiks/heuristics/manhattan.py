########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
########################################################################################################################


class Manhattan(Heuristic):
    """ This heuristic if admissible. The reason is that each non (zero==empty) tile
    has to move by at least its manhattan distance to solve the puzzle, and each move is only moving one tile
    at each step
    """

    @classmethod
    def known_to_be_admissible(cls):
        return True

    def __init__(self, **kw_args):
        kw_args[self.puzzle_type] = Puzzle.sliding_puzzle
        Heuristic.__init__(self, **kw_args)
        # build map of where each value should be
        self.goal_map = {}
        goal = 1
        (self.n, self.m) = self.get_puzzle_dimension()
        size = self.n * self.m
        for row in range(self.n):
            for col in range(self.m):
                self.goal_map[goal] = (row, col)
                goal = goal + 1
                goal %= size

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

