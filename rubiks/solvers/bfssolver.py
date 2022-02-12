########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.search.strategies import BreadthFirstSearch
from rubiks.solvers.solver import Solver
########################################################################################################################


class BFSSolver(Solver):

    def solve_impl(self, puzzle, time_out, **kw_args):
        strat = BreadthFirstSearch(puzzle, time_out=time_out, **{**self.kw_args, **kw_args})
        strat.solve()
        return strat.get_path_cost(), strat.get_path(), strat.get_node_counts()

########################################################################################################################
