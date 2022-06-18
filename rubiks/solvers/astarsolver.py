########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta
########################################################################################################################
from rubiks.search.strategies import AStar
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class AStarSolver(Solver):

    def know_to_be_optimal(self):
        """ unless extremely lucky this is not going to return optimal solutions """
        heuristic = self.kw_args['heuristic']
        return heuristic.known_to_be_admissible()

    def name(self):
        heuristic = self.kw_args['heuristic']
        if hasattr(heuristic, '__class__') and heuristic.__class__ != ABCMeta:
            heuristic = heuristic.__class__
        try:
            heuristic = heuristic(**self.kw_args).name()
        except AttributeError:
            if hasattr(heuristic, '__name__'):
                heuristic = heuristic.__name__
        return '%s|%s|%s' % (self.__class__.__name__,
                             heuristic,
                             self.puzzle_type.construct_puzzle(**self.kw_args).name())

    def solve_impl(self, puzzle, time_out, **kw_args):
        strat = AStar(puzzle, time_out=time_out, **{**self.kw_args, **kw_args})
        strat.solve()
        return Solution(strat.get_path_cost(),
                        strat.get_path(),
                        strat.get_node_counts())

########################################################################################################################
