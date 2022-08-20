########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from itertools import permutations
from math import factorial, inf
from numpy import argwhere, array, where
from pandas import DataFrame
from random import randint
from tabulate import tabulate
from torch import equal, tensor, randperm, reshape, Tensor
from torch.nn.functional import one_hot
########################################################################################################################
from rubiks.puzzle.move import Move
from rubiks.puzzle.puzzle import Puzzle
########################################################################################################################


class Slide(Move):

    def __init__(self, n, m):
        self.tile = (n, m)

    def __eq__(self, other):
        return self.tile == other.tile

    def __ne__(self, other):
        return self.tile != other.tile

    def cost(self):
        return 1

    def __repr__(self):
        return '(%s, %s)' % self.tile

    def __hash__(self):
        return hash((self.__class__.__name__, *self.tile))

########################################################################################################################

    
class SlidingPuzzle(Puzzle):
    """ Game of the sliding Puzzle, e.g. the 8-puzzle, 15-puzzle, etc """

    def number_of_values(self):
        return self.n * self.m

    def number_of_tiles(self):
        return self.n * self.m

    @classmethod
    def nb_moves(cls):
        """ empty might go left, right, up or down """
        return 4

    m = 'm'
    tiles = 'tiles'
    empty = 'empty'

    def empty_left(self):
        return Slide(self.empty[0],
                     self.empty[1] - 1)

    def empty_right(self):
        return Slide(self.empty[0],
                     self.empty[1] + 1)

    def empty_up(self):
        return Slide(self.empty[0] - 1,
                     self.empty[1])

    def empty_down(self):
        return Slide(self.empty[0] + 1,
                     self.empty[1])

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser, field=cls.m, type=int, default=None)
        cls.add_argument(parser, field=cls.tiles, type=list, default=None)

    def possible_puzzles_nb(self):
        dimension = self.dimension()
        try:
            return int(factorial(dimension[0] * dimension[1]) / 2)
        except OverflowError:
            return inf

    @classmethod
    def generate_all_puzzles(cls, **kw_args):
        init = SlidingPuzzle(**kw_args)
        init_signature = init.signature()
        (n, m) = init.dimension()
        for perm in permutations(range(n * m)):
            puzzle = SlidingPuzzle(tiles=tensor(perm).reshape((n, m)))
            if puzzle.signature() == init_signature:
                yield puzzle

    move_type = Slide

    possible_moves_map = dict()

    goal_signature_map = dict()

    goal_map = dict()

    @classmethod
    def __populate_goal__(cls, n, m):
        if n not in cls.goal_map:
            cls.goal_map[n] = dict()
        if m not in cls.goal_map[n]:
            tiles = tensor(range(1, n * m + 1)).reshape((n, m))
            tiles[n - 1][m - 1] = 0
            empty = (n - 1, m - 1)
            cls.goal_map[n][m] = SlidingPuzzle(tiles=tiles, empty=empty)

    def __init__(self, **kw_args):
        from_tiles = self.tiles in kw_args
        if from_tiles and kw_args[self.tiles] is not None:
            # we set n, m and empty
            self.tiles = kw_args[self.tiles]
            if not isinstance(self.tiles, Tensor):
                self.tiles = tensor(self.tiles)
            self.empty = kw_args.get(self.empty,
                                     tuple(argwhere(0 == self.tiles).squeeze().tolist()))
            assert 2 == len(self.tiles.shape)
            kw_args[self.n] = self.tiles.shape[0]
            kw_args[self.m] = self.tiles.shape[1]
            kw_args.pop(__class__.tiles, None)
            kw_args.pop(__class__.empty, None)
            self.puzzle_type = self.sliding_puzzle
            self.n = self.tiles.shape[0]
            self.m = self.tiles.shape[1]
            Puzzle.__init__(self, **kw_args)
        else:
            n = kw_args[self.n]
            m = kw_args.get(self.m, n)
            if m is None:
                m = n
            self.__populate_goal__(n, m) # except the first time this will do nothing
            goal = self.goal_map[n][m]
            self.__init__(tiles=goal.tiles.detach().clone(),
                          empty=goal.empty)

    def find_tile(self, value):
        return tuple(argwhere(value == self.tiles).squeeze().tolist())

    def __repr__(self):
        tiles = DataFrame(self.tiles) #array(self.tiles.numpy(), dtype=str)
        tiles = tiles.stack()
        tiles[tiles == 0] = '\033[38;2;255;165;0m\u2588\u2588\033[0m'
        tiles = tiles.unstack()
        tiles = '\n'.join(tabulate(DataFrame(tiles),
                                   headers='keys',
                                   tablefmt='grid',
                                   showindex=False).split('\n')[2:])
        return '\n' + tiles

    def __eq__(self, other):
        return equal(self.tiles, other.tiles)

    def __hash__(self):
        return hash(tuple(self.tiles.flatten().numpy()))

    def dimension(self):
        return tuple(self.tiles.shape)

    def clone(self):
        return SlidingPuzzle(tiles=self.tiles.detach().clone(),
                             empty=self.empty)

    def is_goal(self):
        return self == self.goal()

    def goal(self):
        self.__populate_goal__(*self.tiles.shape)
        return self.goal_map[self.tiles.shape[0]][self.tiles.shape[1]]

    def apply(self, move: Slide):
        """ moved tile must either be same row or same col as the empty tile 
        and next to it. If they are, we swap empty with slide and return
        """
        mt0 = move.tile[0]
        mt1 = move.tile[1]
        if any(mt < 0 for mt in [mt0, mt1]) or mt0 >= self.tiles.shape[0] or mt1 >= self.tiles.shape[1]:
            raise ValueError('Invalid slide')
        delta_n = mt0 - self.empty[0]
        delta_m = mt1 - self.empty[1]
        if 0 == delta_n:
            if delta_m not in {1, -1}:
                raise ValueError('Invalid slide')
        elif 0 == delta_m:
            if delta_n not in {1, -1}:
                raise ValueError('Invalid slide')
        else:  
            raise ValueError('Invalid slide')
        tiles = self.tiles.detach().clone()
        tiles[self.empty[0]][self.empty[1]] = self.tiles[move.tile[0]][move.tile[1]]
        empty = move.tile
        tiles[empty[0]][empty[1]] = 0
        return SlidingPuzzle(tiles=tiles, empty=empty)

    @staticmethod
    def choices(empty, shape):
        c = []
        if empty[0] > 0:
            c.append(Slide(empty[0] - 1, empty[1]))
        if empty[0] < shape[0] - 1:
            c.append(Slide(empty[0] + 1, empty[1]))
        if empty[1] > 0:
            c.append(Slide(empty[0], empty[1] - 1))
        if empty[1] < shape[1] - 1:
            c.append(Slide(empty[0], empty[1] + 1))
        return c

    @classmethod
    def get_possible_moves(cls, shape):
        possible_moves = {}
        for row in range(shape[0]):
            for col in range(shape[1]):
                empty = (row, col)
                possible_moves[empty] = cls.choices(empty, shape)
        return possible_moves

    @classmethod
    def populate_possible_moves(cls, shape):
        if shape not in cls.possible_moves_map:
            cls.possible_moves_map[shape] = cls.get_possible_moves(shape)

    def possible_moves(self):
        self.populate_possible_moves(self.tiles.shape)
        return self.possible_moves_map[self.tiles.shape][self.empty]

    def theoretical_moves(self):
        return [self.empty_down(), self.empty_left(), self.empty_right(), self.empty_up()]

    def random_move(self):
        self.populate_possible_moves(self.tiles.shape)
        choices = self.possible_moves_map[self.tiles.shape][self.empty]
        if not choices:
            return None
        return choices[randint(0, len(choices) - 1)]

    def from_tensor(self):
        raise NotImplementedError('Please implement this ... need to de-one_hot then call init')
    
    def to_tensor(self, one_hot_encoding=False, flatten=True):
        tiles = self.tiles
        if one_hot_encoding:
            tiles = one_hot(tiles)
        if flatten:
            tiles = tiles.flatten(1)
        return tiles

    def perfect_shuffle(self):
        """ We set up the tiles randomly, and then just swap the first two if the signature is not right """
        dimensions = self.dimension()
        tiles = reshape(randperm(dimensions[0] * dimensions[1]), dimensions)
        for row in range(dimensions[0]):
            if 1 == row % 2:
                tiles[row] = tiles[row].flip(0)
        shuffle = SlidingPuzzle(tiles=tiles)
        if shuffle.signature() == self.goal_signature():
            return shuffle
        row = 0
        col = 0
        while 0 == tiles[row][col]:
            row, col = self.__increment__(row, col)
        row2, col2 = self.__increment__(row, col)
        while 0 == tiles[row2][col2]:
            row2, col2 = self.__increment__(row2, col2)
        a, b = tiles[row][col].item(), tiles[row2][col2].item()
        tiles[row2][col2], tiles[row][col] = a, b
        return SlidingPuzzle(tiles=tiles)

    def __increment__(self, row, col):
        col += 1
        if self.dimension()[1] == col:
            row += 1
            col = 0
        return row, col

    def signature(self) -> int:
        tiles = self.clone()
        dimensions = self.dimension()
        for row in range(dimensions[0]):
            if 1 == row % 2:
                tiles.tiles[row] = tiles.tiles[row].flip(0)
        tiles = tiles.tiles.flatten()
        tiles = tiles[tiles != 0]
        total_unordered = 0
        for index, value in enumerate(tiles):
            total_unordered += sum(tiles[index + 1:] < value)
        return total_unordered.item() % 2

    def goal_signature(self):
        goal = self.goal()
        goal_dim = tuple(goal.dimension())
        if goal_dim not in self.goal_signature_map:
            self.goal_signature_map[goal_dim] = self.goal().signature()
        return self.goal_signature_map[goal_dim]

    def is_solvable(self) -> bool:
        return self.signature() == self.goal_signature()

    def in_order(self):
        """ this is for the more general case where the tiles are not necessarily consecutive integers """
        flat = self.tiles.flatten()
        if 0 not in flat:
            return False
        flat_ordered = flat.sort().values.tolist()[1:]
        return flat_ordered == flat.tolist()[:-1]

    @classmethod
    def hardest(cls, dimension):
        return {(3, 3): [[8, 6, 7], [2, 5, 4], [3, 0, 1]],
                }[dimension]

    @classmethod
    def optimal_solver_config(cls) -> dict:
        from rubiks.solvers.solver import Solver
        from rubiks.heuristics.heuristic import Heuristic
        from rubiks.heuristics.manhattan import Manhattan
        return {Solver.solver_type: Solver.astar,
                Solver.time_out: inf,
                Heuristic.heuristic_type: Heuristic.manhattan,
                Manhattan.plus: True,
                }
    
########################################################################################################################

