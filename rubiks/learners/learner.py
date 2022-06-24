########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
########################################################################################################################
from rubiks.core.parsable import Parsable
from rubiks.utils.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzled
########################################################################################################################


class Learner(Parsable, Puzzled, Loggable, metaclass=ABCMeta):
    """ Generic concept of a learning class that takes a Puzzle and kw_args to construct it, and can:
    - learn to solve the puzzle
    - save/restore its learning to/from a file
    - plot something meaningful from a learning file to show what is going on during the learning process
    """

    learner_type = 'learner_type'
    perfect_learner = 'perfect_learner'
    deep_reinforcement_learner = 'deep_reinforcement_learner'
    known_learner_types = [perfect_learner, deep_reinforcement_learner]

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.learner_type,
                         type=str,
                         default=cls.perfect_learner,
                         choices=cls.known_learner_types)

    @classmethod
    def factory(cls, learner_type, puzzle_type, **kw_args):
        learner_type = str(learner_type).lower()
        kw_args.update({Puzzled.puzzle_type: puzzle_type})
        if any(learner_type.find(what) >= 0 for what in [cls.perfect_learner]):
            from rubiks.learners.perfectlearner import PerfectLearner as LearnerType
        elif any(learner_type.find(what) >= 0 for what in [cls.deep_reinforcement_learner]):
            from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner as LearnerType
        else:
            raise NotImplementedError('Unknown learner_type [%s]' % learner_type)
        return LearnerType(**kw_args)

    def __init__(self, puzzle_type, **kw_args):
        Puzzled.__init__(self, puzzle_type, **kw_args)
        Loggable.__init__(self, log_level=kw_args.pop(Loggable.log_level, Loggable.INFO))

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

    def name(self):
        return '%s|%s' % (self.__class__.__name__,
                          self.puzzle_type.construct_puzzle(**self.kw_args).name())

    @staticmethod
    @abstractmethod
    def plot_learning(learning_file_name,
                      network_name=None,
                      puzzle_type=None,
                      puzzle_dimension=None):
        """ Plot something meaningful. Learning type dependent so will be implemented in derived classes """
        return

########################################################################################################################
