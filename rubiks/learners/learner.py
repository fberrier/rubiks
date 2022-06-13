########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
########################################################################################################################
from rubiks.utils.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzled
########################################################################################################################


class Learner(Puzzled, Loggable, metaclass=ABCMeta):
    """ Generic concept of a learning class that takes a Puzzle and kw_args to construct it, and can:
    - learn to solve the puzzle
    - save/restore its learning to/from a file
    - plot something meaningful from a learning file to show what is going on during the learning process
    """

    def __init__(self, puzzle_type, **kw_args):
        Puzzled.__init__(self, puzzle_type, **kw_args)
        Loggable.__init__(self, self.name(), kw_args.pop('log_level', 'INFO'))

    @abstractmethod
    def learn(self):
        return

    def save(self, model_file_name, **kwargs):
        """ overwrite where meaningful """
        return

    @staticmethod
    def restore(model_file):
        """ overwrite where meaningful """
        return

    def name(self):
        return '%s|%s' %(self.__class__.__name__,
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
