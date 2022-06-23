########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from itertools import permutations
from math import factorial
from numpy import argwhere, array, where
from pandas import DataFrame
from random import randint
from tabulate import tabulate
from torch import equal, tensor, randperm, reshape
from torch.nn.functional import one_hot
########################################################################################################################
from rubiks.puzzle.puzzle import Move, Puzzle
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

########################################################################################################################

    
class SlidingPuzzle(Puzzle):
    """ Game of the sliding Puzzle, e.g. the 8-puzzle, 15-puzzle, etc """

    def possible_puzzles_nb(self):
        dimension = self.dimension()
        return int(factorial(dimension[0] * dimension[1]) / 2)

    @classmethod
    def generate_all_puzzles(cls, **kw_args):
        goal = SlidingPuzzle.construct_puzzle(**kw_args)
        goal_signature = goal.signature()
        (n, m) = goal.dimension()
        for perm in permutations(range(n * m)):
            puzzle = SlidingPuzzle(tensor(perm).reshape((n, m)))
            if puzzle.signature() == goal_signature:
                yield puzzle

    move_type = Slide

    possible_moves_map = {}

    goal_signature_map = {}

    goal_map = {}

    def __init__(self, tiles, empty=None):
        super().__init__()
        self.tiles = tiles
        if empty is None:
            self.empty = tuple(argwhere(0 == tiles).squeeze().tolist())
        else:
            self.empty = empty

    def __repr__(self):
        tiles = array(self.tiles.numpy(), dtype=str)
        tiles = where(tiles == '0', '', tiles)
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
        return self.tiles.shape

    def clone(self):
        return SlidingPuzzle(self.tiles.detach().clone(), self.empty)

    def is_goal(self):
        return self == self.goal()

    @classmethod
    def construct_puzzle(cls, n, m=None, **kw_args):
        if m is None:
            m = n
        goal = tensor(range(1, n * m + 1)).reshape((n, m))
        goal[n - 1][m - 1] = 0
        return SlidingPuzzle(goal, (n - 1, m - 1))

    def goal(self):
        dimension = tuple(self.dimension())
        if dimension not in self.goal_map:
            self.goal_map[dimension] = self.construct_puzzle(*self.tiles.shape)
        return self.goal_map[dimension]

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
        return SlidingPuzzle(tiles, empty)

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

    def random_move(self):
        self.populate_possible_moves(self.tiles.shape)
        choices = self.possible_moves_map[self.tiles.shape][self.empty]
        if not choices:
            return None
        return choices[randint(0, len(choices) - 1)]

    def from_tensor(self):
        raise NotImplementedError('Please implement this ... need to de-one_hot then call init')
    
    def to_tensor(self, one_hot_encoding=False):
        tiles = self.tiles
        if one_hot_encoding:
            tiles = one_hot(tiles)
        return tiles.flatten(1)

    def perfect_shuffle(self):
        """ We set up the tiles randomly, and then just swap the first two if the signature is not right """
        dimensions = self.dimension()
        tiles = reshape(randperm(dimensions[0] * dimensions[1]), dimensions)
        for row in range(dimensions[0]):
            if 1 == row % 2:
                tiles[row] = tiles[row].flip(0)
        shuffle = SlidingPuzzle(tiles)
        if shuffle.signature() == self.goal_signature():
            return shuffle
        row = 0
        col = 0
        while 0 == tiles[row][col]:
            row, col = self.increment(row, col)
        row2, col2 = self.increment(row, col)
        while 0 == tiles[row2][col2]:
            row2, col2 = self.increment(row2, col2)
        a, b = tiles[row][col].item(), tiles[row2][col2].item()
        tiles[row2][col2], tiles[row][col] = a, b
        return SlidingPuzzle(tiles)

    def increment(self, row, col):
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
    
########################################################################################################################
