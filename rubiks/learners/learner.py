########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
########################################################################################################################
from rubiks.core.factory import Factory
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzled import Puzzled
########################################################################################################################


class Learner(Puzzled, Factory, Loggable, metaclass=ABCMeta):
    """ Generic concept of a learning class that takes a Puzzle and kw_args to construct it, and can:
    - learn to solve the puzzle
    - save/restore its learning to/from a file
    - plot something meaningful from a learning file to show what is going on during the learning process
    """

    learner_type = 'learner_type'
    perfect_learner = 'perfect_learner'
    perfect = 'perfect'
    deep_reinforcement_learner = 'deep_reinforcement_learner'
    drl = 'drl'
    known_learner_types = [perfect_learner, deep_reinforcement_learner]

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.learner_type,
                         type=str,
                         default=cls.perfect_learner,
                         choices=cls.known_learner_types)

    @classmethod
    def factory_key_name(cls):
        return cls.learner_type

    @classmethod
    def widget_types(cls):
        from rubiks.learners.perfectlearner import PerfectLearner
        from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
        return {cls.perfect: PerfectLearner,
                cls.drl: DeepReinforcementLearner,
                }

    def __init__(self, **kw_args):
        Puzzled.__init__(self, **kw_args)
        Factory.__init__(self, **kw_args)
        Loggable.__init__(self, **kw_args)

    @abstractmethod
    def learn(self):
        return

    def save(self, model_file_name, **kwargs):
        """ overwrite where meaningful """
        return

    @classmethod
    def restore(cls, model_file):
        """ overwrite where meaningful """
        return

    def get_name(self):
        return '%s|%s' % (self.__class__.__name__, self.puzzle_name())

    @staticmethod
    @abstractmethod
    def plot_learning(learning_file_name,
                      network_name=None,
                      puzzle_type=None,
                      puzzle_dimension=None):
        """ Plot something meaningful. Learning type dependent so will be implemented in derived classes """
        return

########################################################################################################################
