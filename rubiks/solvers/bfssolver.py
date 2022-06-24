########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.search.strategies import BreadthFirstSearch
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class BFSSolver(Solver):

    def know_to_be_optimal(self):
        """ if it does not time out, it is definitely optimal """
        return True

    def solve_impl(self, puzzle, time_out, **kw_args):
        strat = BreadthFirstSearch(puzzle,
                                   time_out=time_out,
                                   **{**self.kw_args, **kw_args})
        strat.solve()
        return Solution(strat.get_path_cost(),
                        strat.get_path(),
                        strat.get_node_counts(),
                        puzzle)

########################################################################################################################
