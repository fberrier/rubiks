########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from os.path import split
########################################################################################################################
from pandas import read_pickle
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.heuristics.heuristic import Heuristic
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.deepqlearner import DeepQLearner
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.learners.deeplearner import DeepLearner
from rubiks.deeplearning.fullyconnected import FullyConnected
from rubiks.deeplearning.convolutional import Convolutional
from rubiks.utils.utils import snake_case, from_snake_case
########################################################################################################################


class DeepLearningHeuristic(Loggable, Heuristic):
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
        self.deep_learning = None
        Loggable.__init__(self, **kw_args)
        Heuristic.__init__(self, **kw_args)
        try:
            if not self.model_file_name:
                self.model_file_name = 'C:/no_file_name_provided.pkl'
                raise ValueError('No model_file_name provided')
            self.deep_learning = DeepLearning.restore(read_pickle(self.model_file_name)
                                                      [DeepReinforcementLearner.network_data_tag])
        except (ValueError, FileNotFoundError) as error:
            error_msg = 'Could not restore DeepLearning model from \'%s\': ' % self.model_file_name
            self.log_warning(error_msg, error)
            self.deep_learning = None

    @staticmethod
    def short_name(model_file_name):
        try:
            long_name = split(model_file_name)[1]
            data = read_pickle(model_file_name)[DeepReinforcementLearner.convergence_data_tag]
            expected_names = [snake_case(DeepQLearner.__name__),
                              snake_case(DeepReinforcementLearner.__name__),
                              snake_case(DeepLearner.__name__)]
            short_name = long_name
            for expected_name in expected_names:
                if long_name.startswith(expected_name):
                    short_name = ''.join([_[0] for _ in expected_name.split('_')])
                    break
            expected_names = [snake_case(FullyConnected.__name__),
                              snake_case(Convolutional.__name__)]
            for expected_name in expected_names:
                if long_name.find(expected_name) >= 0:
                    short_name += '_' + long_name[long_name.find(expected_name):]
                    short_name = short_name[:short_name.find('.')]
                    short_name = short_name.replace(expected_name, ''.join([_[0] for _ in expected_name.split('_')]))
                    break
            short_name += '[puzzles_seen=%.2g%%]' % data[DeepReinforcementLearner.puzzles_seen_pct].iloc[-1]
        except FileNotFoundError:
            return None
        return short_name

    def get_name(self):
        return '%s[%s]' % (Heuristic.get_name(self), split(self.model_file_name)[1])

    def check_puzzle(self, puzzle):
        assert isinstance(puzzle, self.get_puzzle_type_class()), \
            '%s knows cost for %s, not for %s' % (self.__class__.__name__,
                                                  self.puzzle_type,
                                                  puzzle.__class__.__name__)
        assert tuple(puzzle.dimension()) == tuple(self.deep_learning.get_puzzle_dimension()), \
            '%s expected %s of dimension %s, got %s instead' % (self.__class__.__name__,
                                                                puzzle.__class__.__name__,
                                                                tuple(self.deep_learning.get_puzzle_dimension()),
                                                                tuple(puzzle.dimension()))
        if self.deep_learning is None:
            raise RuntimeError\
                ('Was not able to restore DeepLearning model from \'%s\': ' % self.model_file_name)

    def cost_to_go_from_puzzle_impl(self, puzzle):
        self.check_puzzle(puzzle)
        return self.deep_learning.evaluate(puzzle).item()

########################################################################################################################
