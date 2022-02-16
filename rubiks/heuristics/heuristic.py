########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
from torch import Tensor
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
########################################################################################################################


class Heuristic(metaclass=ABCMeta):
    """ Generic concept of a heuristic: a class that is able to give us an estimated cost-to-go for a particular
    type of Puzzle of a given dimension.
    e.g. could be typical heuristics used in A* searches such as those based on Manhattan distance or similar concepts
    or could be neural net based heuristics that have been trained by Deep Reinforcement Learning.
    """

    puzzle_type = int

    @classmethod
    def get_puzzle_type(cls):
        """ returns the type of puzzle that this heuristic deals with """
        assert issubclass(cls.puzzle_type, Puzzle), 'Puzzle type for %s has not been setup properly' % cls.__name__
        return cls.puzzle_type

    def __init__(self, **kw_args):
        """ the kw_args are passed to the underlying type of puzzle that this heuristic deals with """
        self.goal = self.get_puzzle_type().construct_puzzle(**kw_args)

    def puzzle_dimension(self):
        return self.goal.dimension()

    @abstractmethod
    def cost_to_go_from_puzzle_impl(self, puzzle):
        """ this is where derived concrete classes should implement the actual heuristic """
        return

    dim_error_message = 'cost_to_go expected %s of dimension %s. Got %s instead'

    def cost_to_go_from_puzzle(self, puzzle):
        assert puzzle.dimension() == self.puzzle_dimension(), self.dim_error_message % (self.get_puzzle_type(),
                                                                                        self.puzzle_dimension(),
                                                                                        puzzle.dimension())
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

########################################################################################################################