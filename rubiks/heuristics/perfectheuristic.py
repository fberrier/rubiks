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

    @classmethod
    def known_to_be_admissible(cls):
        return True

    def __init__(self, puzzle_type, model_file_name, **kw_args):
        super().__init__(puzzle_type, **kw_args)
        self.data_base = read_pickle(model_file_name)
        puzzle_type = self.data_base[PerfectLearner.puzzle_type_tag]
        dimension = self.data_base[PerfectLearner.dimension_tag]
        assert self.get_puzzle_type() == puzzle_type
        assert dimension == self.puzzle_dimension()
        self.data_base = self.data_base[PerfectLearner.data_tag]

    def cost_to_go_from_puzzle_impl(self, puzzle):
        return self.data_base[hash(puzzle)]

########################################################################################################################

