########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta
########################################################################################################################
from rubiks.search.astarstrategy import AStar
from rubiks.solvers.solver import Solver, Solution
from rubiks.heuristics.heuristic import Heuristic
########################################################################################################################


class AStarSolver(Solver):

    heuristic_type = 'heuristic_type'

    # @todo Francois make sure we can construct the heuristic at init and keep it there rather than
    # reconstructing in solve_impl ... this is idiotic

    def know_to_be_optimal(self):
        """ unless extremely lucky this is not going to return optimal solutions """
        heuristic = Heuristic.factory(**self.get_config())
        return heuristic.known_to_be_admissible()

    def get_name(self):
        return '%s[%s]' % (self.__class__.__name__,
                           Heuristic.factory(**self.get_config()).get_name())

    def solve_impl(self, puzzle, **kw_args):
        strat = AStar(initial_state=puzzle, **kw_args)
        strat.solve()
        return Solution(strat.get_path_cost(),
                        strat.get_path(),
                        strat.get_node_counts(),
                        puzzle=puzzle,
                        solver_name=self.get_name())

########################################################################################################################
