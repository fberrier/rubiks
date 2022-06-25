########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta
########################################################################################################################
from rubiks.search.astarstrategy import AStar
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class AStarSolver(Solver):

    heuristic_type = 'heuristic_type'

    # @todo Francois make sure we can construct the heuristic at init and keep it there rather than
    # reconstructing in solve_impl ... this is idiotic

    def know_to_be_optimal(self):
        """ unless extremely lucky this is not going to return optimal solutions """
        heuristic = self.kw_args[__class__.heuristic_type]
        return heuristic.known_to_be_admissible()

    def get_name(self):
        heuristic = self.kw_args[__class__.heuristic_type]
        if hasattr(heuristic, '__class__') and heuristic.__class__ != ABCMeta:
            heuristic = heuristic.__class__
        try:
            heuristic = heuristic(**self.kw_args).name()
        except AttributeError:
            if hasattr(heuristic, '__name__'):
                heuristic = heuristic.__name__
        return '%s[%s][%s]' % (self.__class__.__name__,
                               heuristic,
                               self.puzzle_name())

    def solve_impl(self, puzzle, **kw_args):
        strat = AStar(initial_state=puzzle, **kw_args)
        strat.solve()
        return Solution(strat.get_path_cost(),
                        strat.get_path(),
                        strat.get_node_counts(),
                        puzzle=puzzle)

########################################################################################################################
