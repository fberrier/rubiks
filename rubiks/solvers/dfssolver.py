########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.search.dfsstrategy import DepthFirstSearch
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class DFSSolver(Solver):

    limit = 'limit'
    default_limit = 100

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         cls.limit,
                         type=int,
                         default=cls.default_limit)

    def __init__(self, puzzle_type, **kw_args):
        Solver.__init__(self,
                        puzzle_type=puzzle_type,
                        **kw_args)

    def know_to_be_optimal(self):
        """ unless extremely lucky this is not going to return optimal solutions """
        return False

    def solve_impl(self, puzzle, **kw_args):
        strat = DepthFirstSearch(puzzle,
                                 **{**self.get_config(), **kw_args})
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

    def get_name(self):
        return self.__class__.__name__ + '[%s=%d]' % (__class__.limit, self.limit)

########################################################################################################################
