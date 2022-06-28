########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from os.path import split
########################################################################################################################
from pandas import read_pickle
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.deeplearning.deeplearning import DeepLearning
########################################################################################################################


class DeepLearningHeuristic(Heuristic):
    """ Just a heuristic that's been learnt by a Deep Learning Network """

    model_file_name = 'model_file_name'

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.model_file_name,
                         type=str,
                         default=None)

    def known_to_be_admissible(self):
        return False

    def __init__(self, model_file_name, **kw_args):
        self.model_file_name = model_file_name
        self.deep_learning = DeepLearning.restore(read_pickle(self.model_file_name)[0])
        Heuristic.__init__(self, **kw_args)

    def get_name(self):
        return '%s[%s]' % (super().get_name(), split(self.model_file_name)[1])

    def cost_to_go_from_puzzle_impl(self, puzzle):
        assert isinstance(puzzle, self.get_puzzle_type_class()), \
            '%s knows cost for %s, not for %s' % (self.__class__.__name__,
                                                  self.puzzle_type,
                                                  puzzle.__class__.__name__)
        assert tuple(puzzle.dimension()) == tuple(self.deep_learning.get_puzzle_dimension()), \
            '%s expected %s of dimension %s, got %s instead' % (self.__class__.__name__,
                                                                puzzle.__class__.__name__,
                                                                tuple(self.deep_learning.get_puzzle_dimension()),
                                                                tuple(puzzle.dimension()))
        return self.deep_learning.evaluate(puzzle)

########################################################################################################################
