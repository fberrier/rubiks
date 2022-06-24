########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.search.strategies import DepthFirstSearch
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class DFSSolver(Solver):

    limit = 'limit'
    default_limit = 100

    @classmethod
    def populate_parser(cls, parser):
        Solver.populate_parser(parser)
        cls.add_argument(parser,
                         'limit',
                         type=int,
                         default=31)


    def __init__(self, puzzle_type, **kw_args):
        kw_args.update({__class__.limit: kw_args.get(__class__.limit,
                                                     __class__.default_limit),
                        })
        Solver.__init__(self,
                        puzzle_type=puzzle_type,
                        **kw_args)

    def know_to_be_optimal(self):
        """ unless extremely lucky this is not going to return optimal solutions """
        return False

    def solve_impl(self, puzzle, time_out, **kw_args):
        strat = DepthFirstSearch(puzzle, time_out=time_out, **{**self.kw_args, **kw_args})
        strat.solve()
        return Solution(strat.get_path_cost(),
                        strat.get_path(),
                        strat.get_node_counts(),
                        puzzle)

    def name(self):
        return self.__class__.__name__ + '[%s=%d]' % (__class__.limit,
                                                      self.kw_args.get(__class__.limit))

########################################################################################################################
