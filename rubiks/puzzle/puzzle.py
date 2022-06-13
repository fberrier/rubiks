########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
from math import factorial, isinf
from numpy import prod
from numpy.random import permutation
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
                          min_no_loop=1,
                          one_list=False,
                          **kw_args):
        """
        Produces training data.
        params:
            nb_shuffles: number of shuffles we do from goal state
            nb_sequences: how many such sequences we produce
            min_no_loop: (best effort) making sequences with no loop of length <= min_no_loop
            one_list: results is a list, otherwise a list of lists
        returns:
            list of puzzles
        """
        goal = cls.construct_puzzle(**kw_args)
        nb_shuffles = min(nb_shuffles, factorial(prod(goal.dimension())) / 2)
        training_data = []
        for _ in range(nb_sequences):
            moves = goal.random_moves(nb_shuffles, min_no_loop=min_no_loop)
            puzzles = goal.get_puzzle_sequence(moves)
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

    def get_puzzle_sequence(self, moves):
        puzzles = [self.clone()]
        for move in moves:
            puzzles.append(puzzles[-1].apply(move))
        return puzzles

    def apply_moves(self, moves):
        puzzle = self.clone()
        for move in moves:
            puzzle = puzzle.apply(move)
        return puzzle

    @abstractmethod
    def random_move(self):
        """ return a random move from current state """
        return

    def random_moves(self, nb_moves, min_no_loop=1):
        """ return a sequence of r random moves from current state.
        If possible choosing moves that do not create a cycle of length <= min_no_loop """
        assert not isinf(nb_moves)
        puzzle = self.clone()
        last_puzzles = [hash(puzzle)] * min_no_loop
        moves = []
        for _ in range(nb_moves):
            possible_moves = puzzle.possible_moves()
            possible_moves = [possible_moves[p] for p in permutation(range(len(possible_moves)))]
            hash_puzzle = None
            for move in possible_moves:
                candidate = puzzle.apply(move)
                hash_puzzle = hash(candidate)
                if hash(candidate) not in last_puzzles:
                    break
            puzzle = candidate
            last_puzzles = last_puzzles[1:] + [hash_puzzle]
            moves.append(move)
        return moves

    @abstractmethod
    def possible_moves(self) -> list:
        """ return the set of possible moves from this configuration """
        return

    very_large_nb_shuffle = 10000

    def perfect_shuffle(self):
        """ We apply a large number of shuffles, but this can be over-ridden in derived classes
        if there is a known way to actually get a perfectly shuffled instance of the puzzle in question.
        """
        return self.apply_random_moves(self.very_large_nb_shuffle)

    def apply_random_move(self):
        random_move = self.random_move()
        if random_move is None:
            return self.clone()
        return self.apply(random_move)

    def apply_random_moves(self, nb_moves, min_no_loop=1):
        if isinf(nb_moves):
            return self.perfect_shuffle()
        return self.apply_moves(self.random_moves(nb_moves=nb_moves,
                                                  min_no_loop=min_no_loop))

    @abstractmethod
    def to_tensor(self) -> Tensor:
        """ return a torch.Tensor to represent internal state """
        return

########################################################################################################################


class Puzzled:

    def __init__(self, puzzle_type, **kw_args):
        self.puzzle_type = puzzle_type
        self.kw_args = kw_args
        self.goal = self.puzzle_type.construct_puzzle(**self.kw_args)

    def get_puzzle_type(self):
        """ returns the type of puzzle that this heuristic deals with """
        assert issubclass(self.puzzle_type,
                          Puzzle), 'Puzzle type for %s has not been setup properly' % self.__class__.__name__
        return self.puzzle_type

    def puzzle_dimension(self):
        return self.goal.dimension()

    def get_goal(self):
        return self.goal.clone()

    def puzzle_name(self):
        return self.goal.name()

########################################################################################################################
