########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
########################################################################################################################
from rubiks.core.factory import Factory
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzled import Puzzled
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.utils.utils import remove_file
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
    action_type = 'action_type'
    do_plot = 'do_plot'
    do_learn = 'do_learn'
    do_cleanup_learning_file = 'do_cleanup_learning_file'
    known_action_type = [do_learn, do_plot, do_cleanup_learning_file]
    learning_file_name = 'learning_file_name'

    @classmethod
    def additional_dependencies(cls):
        return [DeepLearning] + list(DeepLearning.widget_types().values())

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.learner_type,
                         type=str,
                         default=cls.perfect_learner,
                         choices=cls.known_learner_types)
        cls.add_argument(parser,
                         field=cls.learning_file_name,
                         type=str)
        cls.add_argument(parser,
                         cls.action_type,
                         type=str,
                         default=False,
                         choices=cls.known_action_type)

    def action(self):
        if self.do_plot == self.action_type:
            self.plot_learning()
        elif self.do_learn == self.action_type:
            self.learn()
        elif self.do_cleanup_learning_file == self.action_type:
            self.cleanup_learning_file()
        else:
            raise NotImplementedError('Unknown action_type [%s]' % self.action_type)

    def cleanup_learning_file(self):
        try:
            remove_file(self.learning_file_name)
            self.log_info('Removed \'%s\'' % self.learning_file_name)
        except FileNotFoundError:
            pass

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
    def restore(cls, model_file_name):
        """ overwrite where meaningful """
        return

    def get_name(self):
        return '%s|%s' % (self.__class__.__name__, self.puzzle_name())

    @abstractmethod
    def plot_learning(self):
        """ Plot something meaningful. Learning type dependent so will be implemented in derived classes """
        return

########################################################################################################################
