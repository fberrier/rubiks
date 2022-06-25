########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
########################################################################################################################


class Move(metaclass=ABCMeta):
    """ Generic concept of move """

    @abstractmethod
    def __eq__(self, other):
        return

    @abstractmethod
    def __ne__(self, other):
        return

    @abstractmethod
    def cost(self):
        """ What is the cost of this move """
        return


########################################################################################################################

