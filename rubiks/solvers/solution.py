########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf, isinf
from pandas import Series
########################################################################################################################
from rubiks.utils.utils import pformat
########################################################################################################################


class Solution:

    cost = 'cost'
    path = 'path'
    expanded_nodes = '# expanded nodes'
    puzzle = 'puzzle'

    def apply(self, puzzle):
        return puzzle.apply_moves(self.path)

    def __init__(self,
                 cost,
                 path,
                 expanded_nodes,
                 puzzle=None,
                 **additional_info):
        self.cost = cost
        self.path = path
        self.expanded_nodes = expanded_nodes
        self.puzzle = puzzle
        self.additional_info = additional_info

    @classmethod
    def failure(cls, puzzle, **additional_info):
        return Solution(inf,
                        list(),
                        inf,
                        puzzle=puzzle,
                        **additional_info)

    def failed(self):
        return isinf(self.cost)

    def to_str(self):
        cls = self.__class__
        if self.puzzle is None:
            raise ValueError('Cannot convert solution to string as missing initial puzzle')
        puzzles = [str(self.puzzle)] if not isinf(self.cost) else []
        puzzle = self.puzzle.clone()
        for move in self.path:
            puzzle = puzzle.apply(move)
            puzzles.append(str(puzzle))
        path_string = '\n' + pformat(Series(index=range(len(puzzles)),
                                            data=puzzles,
                                            dtype=str))
        string = {cls.puzzle: self.puzzle,
                  cls.cost: self.cost,
                  cls.expanded_nodes: self.expanded_nodes,
                  cls.path: path_string,
                  **{'%s' % k: '%s' % v for k, v in self.additional_info.items()},
                  }
        return '\n' + pformat(string)

    def __repr__(self):
        return self.to_str()

########################################################################################################################
