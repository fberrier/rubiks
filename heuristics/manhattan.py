########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from heuristics.heuristics import Heuristic
from rubiks.puzzle.sliding import SlidingPuzzle
########################################################################################################################


class Manhattan(Heuristic):
    """ TBD """

    puzzle_type = SlidingPuzzle


    def __init__(self, n, m=None, verbose=False):
        super().__init__(n=n, m=m)
        # build map of where each value should be
        self.goal_map = {}
        goal = 1
        size = n*m
        for row in range(n):
            for col in range(m):
                self.goal_map[goal] = (row, col)
                goal = goal + 1
                goal %= size
        (self.n, self.m) = tuple(self.puzzle_dimension())
        self.verbose = verbose

    def debug(self, *what):
        if self.verbose:
            print(*what)

    def cost_to_go_from_puzzle_impl(self, puzzle):
        self.debug('cost of ', puzzle)
        puzzle = puzzle.clone()
        cost_to_go = 0
        row = 0
        col = 0
        max_cost = 0
        while row < self.n:
            self.debug('#####')
            self.debug('row:', row)
            self.debug('col:', col)
            val = puzzle.tiles[row][col].item()
            goal = self.goal_map[val]
            cost = abs(row - goal[0]) + abs(col - goal[1])
            self.debug('val ', val, ' should be in position ', goal, ' at distance ', cost)
            if cost > 0:
                # swap
                puzzle.tiles[row][col] = puzzle.tiles[goal[0]][goal[1]]
                puzzle.tiles[goal[0]][goal[1]] = val
                # stay there
                max_cost = max(cost, max_cost)
                self.debug('after move: ', puzzle)
                self.debug('cost: ', cost)
                self.debug('max_cost: ', max_cost)
                self.debug('cost_to_go: ', cost_to_go)
            else:
                # move on
                self.debug('ok')
                col += 1
                if col == self.m:
                    col = 0
                    row += 1
                cost_to_go += max_cost
                max_cost = 0
        self.debug('########')
        self.debug('cost_to_go: ', cost_to_go)
        return cost_to_go

    

########################################################################################################################
