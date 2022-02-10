########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
########################################################################################################################


class Move(metaclass=ABCMeta):
    """ Generic concept of move """
    pass

########################################################################################################################


class Puzzle(metaclass=ABCMeta):
    """ Generic concept of a puzzle """

    move_type = int

    @classmethod
    def get_move_type(cls):
        assert issubclass(cls.move_type, Move), 'Move type for %s has not been setup properly' % cls.__name__
        return cls.move_type

    @abstractmethod
    def goal_state(self):
        """ return the unique goal state for that puzzle """
        return

    @abstractmethod
    def apply(self, move):
        """ return the puzzle resulting from applying move or raise exception if invalid """
        return

    @abstractmethod
    def random_move(self):
        """ return a random move from this state """
        return

    def apply_random_move(self):
        return self.apply(self.random_move())

########################################################################################################################
