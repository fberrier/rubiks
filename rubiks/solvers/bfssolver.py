########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.search.bfsstrategy import BreadthFirstSearch
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class BFSSolver(Solver):

    def known_to_be_optimal(self):
        """ if it does not time out, it is definitely optimal """
        return True

    def solve_impl(self, puzzle, **kw_args):
        strat = BreadthFirstSearch(puzzle, **{**self.get_config(), **kw_args})
        try:
            strat.solve()
        except TimeoutError:
            solution = Solution.failure(puzzle,
                                        time_out=True,
                                        failure_reson='time out')
            solution.expanded_nodes = strat.get_node_counts()
            return solution
        return Solution(strat.get_path_cost(),
                        strat.get_path(),
                        strat.get_node_counts(),
                        puzzle,
                        solver_name=self.get_name())

########################################################################################################################
