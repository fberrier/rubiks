########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
from torch import Size, stack, tensor, Tensor
########################################################################################################################


class Move(metaclass=ABCMeta):
    """ Generic concept of move """
    pass

########################################################################################################################


class Puzzle(metaclass=ABCMeta):
    """ Generic concept of a puzzle """

    move_type = int

    @abstractmethod
    def dimension(self) -> Size:
        """ returns the dimension of that puzzle. The concept of dimension is type of puzzle dependent obviously """
        return

    @classmethod
    def get_move_type(cls):
        assert issubclass(cls.move_type, Move), 'Move type for %s has not been setup properly' % cls.__name__
        return cls.move_type

    @classmethod
    def get_puzzle(cls, nb_shuffle, **kw_args):
        """ return a puzzle shuffled nb_shuffle times randomly from goal state
        kw_args allow to construct the puzzle with e.g. puzzle dimension
        """
        puzzle = cls.construct_puzzle(**kw_args)
        for _ in range(nb_shuffle):
            puzzle = puzzle.apply_random_move()
        return puzzle
    
    @abstractmethod
    def clone(self):
        return

    @abstractmethod
    def is_goal(self) -> bool:
        """ is this puzzle the goal state? """
        return

    @classmethod
    def get_training_data(cls, nb_shuffle, nb_sequences, **kw_args) -> (Tensor, Tensor):
        """
        :param nb_shuffle: number of shuffles we do from goal state
        :param nb_sequences: how many such sequences we produce
        :param kw_args: args to be passed to constructor of the puzzle
        :returns: (tensor of puzzles, tensor of nb shuffles)
        """
        goal = cls.construct_puzzle(**kw_args)
        training_data = []
        for _ in range(nb_sequences):
            puzzle = goal.clone()
            training_data.append(puzzle.to_tensor())
            for __ in range(nb_shuffle):
                puzzle = puzzle.apply_random_move()
                training_data.append(puzzle.to_tensor())
        return (stack(training_data), tensor(range(nb_shuffle + 1)).repeat(nb_sequences))    

    @classmethod
    @abstractmethod
    def construct_puzzle(cls, **kw_args):
        """ construct a puzzle according to specification given by kw_args """
        return

    @classmethod
    @abstractmethod
    def from_tensor(cls, data, **kw_args):
        """ construct a puzzle from its tensor representation """
        return

    @abstractmethod
    def goal(self):
        """ return the unique goal state for that puzzle """
        return

    @abstractmethod
    def apply(self, move):
        """ return the puzzle resulting from applying move or raise exception if invalid """
        return

    @abstractmethod
    def random_move(self):
        """ return a random move from current state """
        return

    @abstractmethod
    def possible_moves(self) -> set:
        """ return the set of possible moves from this configuration """
        return

    def apply_random_move(self):
        random_move = self.random_move()
        if random_move is None:
            return self.clone()
        return self.apply(random_move)

    @abstractmethod
    def to_tensor(self) -> Tensor:
        """ return a torch.Tensor to represent internal state """
        return

########################################################################################################################
