########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.deeplearning.deeplearning import DeepLearning
########################################################################################################################


class DeepLearningHeuristic(Heuristic):
    """ TBD """

    def __init__(self, model_file, **kw_args):
        self.deep_learning = DeepLearning.restore(model_file)
        self.puzzle_type = self.deep_learning.puzzle_type
        Heuristic.__init__(self, **kw_args)

    def cost_to_go_from_puzzle_impl(self, puzzle):
        assert isinstance(puzzle, self.deep_learning.puzzle_type), \
            '%s knows cost for %s, not for %s' (self.__class__.__name__,
                                                self.deep_learning.puzzle_type,
                                                puzzle.__class__.__name__)
        assert puzzle.dimension() == self.deep_learning.puzzle_dimension, \
            '%s expected %s of dimension %s, for %s instead' % (self.__class__.__name__,
                                                                puzzle.__class__.__name__,
                                                                self.deep_learning.puzzle_dimension,
                                                                puzzle.dimension())
        return self.deep_learning.evaluate(puzzle)

########################################################################################################################
