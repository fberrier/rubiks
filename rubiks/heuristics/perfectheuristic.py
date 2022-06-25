########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from pandas import read_pickle
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.learners.perfectlearner import PerfectLearner
########################################################################################################################


class PerfectHeuristic(Heuristic):
    """ This is used only for small puzzles where we can literally run all possible puzzles of
    a given dimension and solve them via an optimal solver, and save that down to a file
    """

    model_file_name = 'model_file_name'

    @classmethod
    def known_to_be_admissible(cls):
        return True

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.model_file_name,
                         type=str,
                         default=None)

    def __init__(self, **kw_args):
        Heuristic.__init__(self, **kw_args)
        self.data_base = read_pickle(self.model_file_name)
        puzzle_type = self.data_base[PerfectLearner.puzzle_type]
        dimension = self.data_base[PerfectLearner.dimension]
        assert self.get_puzzle_type() == puzzle_type
        assert dimension == self.get_puzzle_dimension()
        self.data_base = self.data_base[PerfectLearner.data]

    def cost_to_go_from_puzzle_impl(self, puzzle):
        return self.data_base[hash(puzzle)]

########################################################################################################################

