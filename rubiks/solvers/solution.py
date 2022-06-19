########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from pandas import Series
########################################################################################################################
from rubiks.utils.utils import pformat
########################################################################################################################


class Solution:

    def __init__(self,
                 cost,
                 path,
                 expanded_nodes,
                 puzzle=None):
        self.cost = cost
        self.path = path
        self.expanded_nodes = expanded_nodes
        self.puzzle = puzzle

    def to_str(self, puzzle=None):
        if puzzle is None:
            puzzle = self.puzzle
        if puzzle is None:
            raise ValueError('Cannot convert solution to string as missing initial puzzle')
        puzzles = [str(puzzle)]
        for move in self.path:
            puzzle = puzzle.apply(move)
            puzzles.append(str(puzzle))
        return '\n' + pformat(Series(index=range(len(puzzles)),
                                     data=puzzles))

    def __repr__(self):
        return self.to_str()

########################################################################################################################
