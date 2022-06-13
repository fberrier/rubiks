########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.search.strategies import DepthFirstSearch
from rubiks.solvers.solver import Solver
########################################################################################################################


class DFSSolver(Solver):

    def know_to_be_optimal(self):
        """ unless extremely lucky this is not going to return optimal solutions """
        return False

    def solve_impl(self, puzzle, time_out, **kw_args):
        strat = DepthFirstSearch(puzzle, time_out=time_out, **{**self.kw_args, **kw_args})
        strat.solve()
        return strat.get_path_cost(), strat.get_path(), strat.get_node_counts()

########################################################################################################################
