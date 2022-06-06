########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
from math import factorial, isinf
from numpy import prod
from torch import Size, Tensor
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


class Puzzle(metaclass=ABCMeta):
    """ Generic concept of a puzzle """

    move_type = int

    @abstractmethod
    def dimension(self) -> Size:
        """ returns the dimension of that puzzle. The concept of dimension is type of puzzle dependent obviously """
        return

    @abstractmethod
    def __hash__(self):
        """ Useful to put in containers that rely on hashes """
        pass
    
    @abstractmethod
    def __eq__(self, other):
        pass

    @classmethod
    def get_move_type(cls):
        assert issubclass(cls.move_type, Move), 'Move type for %s has not been setup properly' % cls.__name__
        return cls.move_type

    @classmethod
    def get_puzzle(cls, nb_shuffles, **kw_args):
        """ return a puzzle shuffled nb_shuffles times randomly from goal state
        kw_args allow to construct the puzzle with e.g. puzzle dimension
        """
        puzzle = cls.construct_puzzle(**kw_args)
        for _ in range(nb_shuffles):
            puzzle = puzzle.apply_random_move()
        return puzzle
    
    @abstractmethod
    def clone(self):
        return

    def name(self):
        return '%s[%s]' % (self.__class__.__name__, str(tuple(self.dimension())))

    @abstractmethod
    def is_goal(self) -> bool:
        """ is this puzzle the goal state? """
        return

    @classmethod
    def get_training_data(cls,
                          nb_shuffles,
                          nb_sequences,
                          strict=True,
                          one_list=False,
                          **kw_args):
        """
        Produces training data.
        params:
            nb_shuffles: number of shuffles we do from goal state
            nb_sequences: how many such sequences we produce
            strict: make sure it s actually different puzzles for each sequence
            one_list: results is a list, otherwise a list of lists
        returns:
            list of puzzles
        """
        goal = cls.construct_puzzle(**kw_args)
        max_size = factorial(prod(goal.dimension())) / 2
        training_data = []
        hashes = set()
        for _ in range(nb_sequences):
            puzzles = []
            puzzle = goal.clone()
            puzzles.append(puzzle)
            if strict:
                hashes.clear()
                hashes.add(hash(puzzle))
            for __ in range(nb_shuffles):
                puzzle = puzzle.apply_random_move()
                puzzle_hash = hash(puzzle)
                while strict and puzzle_hash in hashes:
                    puzzle = puzzle.apply_random_move()
                    puzzle_hash = hash(puzzle)                    
                hashes.add(puzzle_hash)
                puzzles.append(puzzle)
                if len(puzzles) >= max_size:
                    break
            while len(puzzles) < nb_shuffles:
                puzzles.append(puzzles[-1])
            if one_list:
                training_data += puzzles
            else:
                training_data.append(puzzles)
        return training_data

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

    very_large_nb_shuffle = 10000

    def perfect_shuffle(self):
        """ We apply a large number of shuffles, but this can be over-riden in derived classes
        if there is a known way to actually get a perfectly shuffled instance of the puzzle in question.
        """
        return self.apply_random_moves(self.very_large_nb_shuffle)

    def apply_random_move(self):
        random_move = self.random_move()
        if random_move is None:
            return self.clone()
        return self.apply(random_move)

    def apply_random_moves(self, r):
        if isinf(r):
            return self.perfect_shuffle()
        move = self.clone()
        for _ in range(r):
            move = move.apply_random_move()
        return move

    @abstractmethod
    def to_tensor(self) -> Tensor:
        """ return a torch.Tensor to represent internal state """
        return

########################################################################################################################
