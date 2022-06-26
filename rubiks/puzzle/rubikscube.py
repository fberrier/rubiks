########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy import argwhere, array, where
from pandas import DataFrame
from random import randint
from tabulate import tabulate
from torch import equal, tensor, randperm, reshape
from torch.nn.functional import one_hot
########################################################################################################################
from rubiks.puzzle.puzzle import Move, Puzzle
########################################################################################################################


class CubeMove(Move):

    def __init__(self):
        pass

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return False

    def cost(self):
        return 1

########################################################################################################################

    
class RubiksCube(Puzzle):
    """ Game of the sliding Puzzle, e.g. the 8-puzzle, 15-puzzle, etc """

    @classmethod
    def generate_all_puzzles(cls, **kw_args):
        pass

    move_type = CubeMove

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return

    def __eq__(self, other):
        return False

    def __hash__(self):
        return 0

    def dimension(self):
        return 0

    def clone(self):
        return 0

    def is_goal(self):
        return False

    @classmethod
    def construct_puzzle(cls, n, **kw_args):
        return

    def goal(self):
        return

    def apply(self, move: CubeMove):
        return self.clone()

    def possible_moves(self):
        return

    def random_move(self):
        return

    def from_tensor(self):
        raise NotImplementedError('Please implement this ... need to de-one_hot then call init')
    
    def to_tensor(self, one_hot_encoding=False):
        raise NotImplementedError('Please implement this ... RubiksCube.to_tensor')

    def perfect_shuffle(self):
        return
    
########################################################################################################################
