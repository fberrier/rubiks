########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
########################################################################################################################
from rubiks.utils.loggable import Loggable
########################################################################################################################


class Learner(Loggable, metaclass=ABCMeta):
    """ TBD """

    def __init__(self, puzzle_type, **kw_args):
        self.puzzle_type = puzzle_type
        self.kw_args = kw_args
        Loggable.__init__(self, self.name(), kw_args.pop('log_level', 'INFO'))

    @abstractmethod
    def learn(self):
        return

    def save(self, data_base):
        """ overwrite where meaningful """
        return

    @staticmethod
    def restore(data_base):
        """ overwrite where meaningful """
        return

    def name(self):
        return '%s|%s' %(self.__class__.__name__,
                         self.puzzle_type.construct_puzzle(**self.kw_args).name())

########################################################################################################################
