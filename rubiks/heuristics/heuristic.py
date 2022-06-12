########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
from torch import Tensor
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzled
########################################################################################################################


class Heuristic(Puzzled, metaclass=ABCMeta):
    """ Generic concept of a heuristic: a class that is able to give us an estimated cost-to-go for a particular
    type of Puzzle of a given dimension.
    e.g. could be typical heuristics used in A* searches such as those based on Manhattan distance or similar concepts
    or could be neural net based heuristics that have been trained by Deep Reinforcement Learning.
    """
    def __init__(self, puzzle_type, **kw_args):
        """ the kw_args are passed to the underlying type of puzzle that this heuristic deals with """
        Puzzled.__init__(self, puzzle_type, **kw_args)

    @abstractmethod
    def cost_to_go_from_puzzle_impl(self, puzzle):
        """ this is where derived concrete classes should implement the actual heuristic """
        return

    dim_error_message = 'cost_to_go expected %s of dimension %s. Got %s instead'

    def cost_to_go_from_puzzle(self, puzzle):
        assert puzzle.dimension() == self.puzzle_dimension(), self.dim_error_message % (str(self.get_puzzle_type()),
                                                                                        str(self.puzzle_dimension()),
                                                                                        str(puzzle.dimension()))
        return self.cost_to_go_from_puzzle_impl(puzzle)

    def cost_to_go_from_tensor(self, puzzle, **kw_args):
        """ feel free to overwrite this by more efficient in derived classes. If not, we convert to 
        puzzle from tensor from the function that the puzzle (has to) provide
        """
        return self.cost_to_go_from_puzzle(self.puzzle_type.from_tensor(puzzle, **kw_args))

    def cost_to_go(self, puzzle, **kw_args):
        if isinstance(puzzle, Tensor):
            return self.cost_to_go_from_tensor(puzzle, **kw_args)
        return self.cost_to_go_from_puzzle(puzzle)

    def name(self):
        return self.__class__.__name__

########################################################################################################################

