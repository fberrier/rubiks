########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf, isinf
from pandas import Series, DataFrame
########################################################################################################################
from rubiks.utils.utils import pformat, number_format
########################################################################################################################


class Solution:

    cost = 'cost'
    path = 'path'
    expanded_nodes = '# expanded nodes'
    puzzle = 'puzzle'
    success = 'success'
    solver_name = 'solver_name'
    run_time = 'run_time'

    def apply(self, puzzle):
        return puzzle.apply_moves(self.path)

    def __init__(self,
                 cost,
                 path,
                 expanded_nodes,
                 puzzle=None,
                 success=True,
                 time_out=False,
                 run_time=float('nan'),
                 **additional_info):
        self.cost = cost
        self.path = path
        self.expanded_nodes = expanded_nodes
        self.puzzle = puzzle
        self.success = success
        self.time_out = time_out
        self.run_time = run_time
        self.additional_info = additional_info

    def set_run_time(self, run_time):
        self.run_time = run_time

    def set_additional_info(self, **kw_args):
        self.additional_info.update(kw_args)

    @classmethod
    def failure(cls, puzzle, **additional_info):
        return Solution(inf,
                        list(),
                        inf,
                        puzzle=puzzle,
                        success=False,
                        **additional_info)

    def failed(self):
        return isinf(self.cost) or not self.success

    def add_additional_info(self, **additional_info):
        self.additional_info.update(additional_info)

    def to_str(self, fields=None):
        cls = self.__class__
        if self.puzzle is None:
            raise ValueError('Cannot convert solution to string as missing initial puzzle')
        puzzles = [str(self.puzzle)] if not isinf(self.cost) else []
        puzzle = self.puzzle.clone()
        for move in self.path:
            puzzle = puzzle.apply(move)
            puzzles.append(str(puzzle))
        string = {cls.puzzle: self.puzzle,
                  cls.cost: self.cost,
                  cls.expanded_nodes: number_format(self.expanded_nodes),
                  cls.success: 'Y' if self.success else 'N',
                  **{'%s' % k: '%s' % v for k, v in self.additional_info.items()},
                  }
        for m_nb, (move, puzzle) in enumerate(zip(['Start'] + self.path, puzzles)):
            string['%s%s' % ('' if m_nb == 0 else 'Move %d -- ' % (m_nb + 1), move)] = puzzle
        string_fields = list(string.keys())
        for field in string_fields:
            if fields is not None and field not in fields:
                string.pop(field, None)
        return '\n' + pformat(string)

    def to_str_light(self):
        return self.to_str(fields=[Solution.puzzle,
                                   Solution.cost,
                                   Solution.expanded_nodes,
                                   Solution.success,
                                   Solution.solver_name,
                                   Solution.run_time])

    def __repr__(self):
        return self.to_str()

    def get_path(self):
        puzzles = [self.puzzle]
        for move in self.path:
            puzzle = puzzles[-1].apply(move)
            puzzles.append(puzzle)
        return puzzles

    def __len__(self):
        return self.cost

########################################################################################################################
