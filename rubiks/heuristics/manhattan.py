########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from os.path import isfile
from pandas import read_pickle
from itertools import permutations
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.utils.utils import get_file_name, Extension, PossibleFileNames, to_pickle
########################################################################################################################


class Manhattan(Heuristic):
    """ This heuristic if admissible. The reason is that each non (zero==empty) tile
    has to move by at least its manhattan distance to solve the puzzle, and each move is only moving one tile
    at each step
    """

    def get_name(self):
        class_name = self.__class__.__name__
        if self.plus:
            class_name += '++'
        return class_name

    @classmethod
    def get_manhattan_plus_file_name(cls, dimension):
        return get_file_name(Puzzle.sliding_puzzle,
                             dimension,
                             PossibleFileNames.manhattan,
                             extension=Extension.pkl,
                             name=PossibleFileNames.manhattan)

    plus = 'plus'

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.plus,
                         default=False,
                         action=cls.store_true)

    @classmethod
    def known_to_be_admissible(cls):
        return True

    def __init__(self, **kw_args):
        """ Specific heuristic for the sliding_puzzle """
        Heuristic.__init__(self, **{**kw_args, self.puzzle_type: Puzzle.sliding_puzzle})
        # build map of where each value should be
        self.goal_map = {}
        goal = 1
        dimension = self.get_puzzle_dimension()
        (self.n, self.m) = dimension
        size = self.n * self.m
        for row in range(self.n):
            for col in range(self.m):
                self.goal_map[goal] = (row, col)
                goal = goal + 1
                goal %= size
        if self.plus:
            manhattan_plus_file_name = self.get_manhattan_plus_file_name(dimension)
            if not isfile(manhattan_plus_file_name):
                self.pre_compute_linear_constraints(*dimension)
            self.plus_data = read_pickle(manhattan_plus_file_name)

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
        if self.plus:
            data = self.plus_data[self.Line.row]
            for row in range(self.n):
                if row in data:
                    line = tuple(puzzle.tiles[row].tolist())
                    if line in data[row]:
                        cost_to_go += data[row][line]
            data = self.plus_data[self.Line.col]
            for col in range(self.m):
                if col in data:
                    line = tuple(puzzle.tiles[:, col].tolist())
                    if line in data[col]:
                        cost_to_go += data[col][line]
        return cost_to_go

    class Line:
        row = 'row'
        col = 'col'

    @classmethod
    def penalty(cls, expected, actual):
        es = set(expected)
        actual = [a for a in actual if 0 != a and a in expected]
        if len(actual) <= 1:
            return 0
        if len(actual) == 2:
            return 2 * (actual[0] > actual[1])
        m = min(actual)
        M = max(actual)
        to_compare = list()
        to_compare.append(2 * (m != actual[0]) + cls.penalty(expected, actual[1:]))
        to_compare.append(2 * (M != actual[-1]) + cls.penalty(expected, actual[:-1]))
        return min(to_compare)

    @classmethod
    def pre_compute_linear_constraints(cls, n, m):
        dimension = (n, m)
        logger = Loggable(name='pre_compute_linear_constraints(%d, %d)' % dimension)
        manhattan_plus_file_name = cls.get_manhattan_plus_file_name(dimension)
        """ There are n rows, each can have (n * m)!/(n * m - m)! values
                      m columns, each can have (n * m)!/(n * m - n)! values
        Better get going!
        """
        data = dict()
        data[cls.Line.row] = dict()
        data[cls.Line.col] = dict()
        goal = Puzzle.factory(puzzle_type=Puzzle.sliding_puzzle,
                              n=n,
                              m=m)
        total_precomputations = 0
        for row in range(n):
            expected = tuple(goal.tiles[row].tolist())
            for possible in permutations(range(n * m), m):
                possible = tuple(possible)
                penalty = cls.penalty(expected, possible)
                if penalty > 0:
                    if row not in data[cls.Line.row]:
                        data[cls.Line.row][row] = dict()
                    data[cls.Line.row][row][possible] = penalty
                    logger.log_info(cls.Line.row, ' ', row, ' ', possible, ' -> ', penalty)
                total_precomputations += 1
        for col in range(m):
            expected = tuple(goal.tiles[:, col].tolist())
            for possible in permutations(range(n * m), n):
                possible = tuple(possible)
                penalty = cls.penalty(expected, possible)
                if penalty > 0:
                    if col not in data[cls.Line.col]:
                        data[cls.Line.col][col] = dict()
                    data[cls.Line.col][col][possible] = penalty
                    logger.log_info(cls.Line.col, ' ', col, ' ', possible, ' -> ', penalty)
                total_precomputations += 1
        to_pickle(data, manhattan_plus_file_name)
        logger.log_info('Saved ',
                        total_precomputations,
                        'pre-computations in',
                        manhattan_plus_file_name)

########################################################################################################################

