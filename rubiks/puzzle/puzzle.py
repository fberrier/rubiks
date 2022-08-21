########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import ABCMeta, abstractmethod
from math import inf
from numpy.random import permutation
from torch import Size, Tensor
from types import MethodType
########################################################################################################################
from rubiks.core.factory import Factory
from rubiks.puzzle.move import Move
from rubiks.utils.utils import is_inf
########################################################################################################################


class Puzzle(Factory, metaclass=ABCMeta):
    """ Generic concept of a puzzle """

    move_type = int
    puzzle_type = 'puzzle_type'
    sliding_puzzle = 'sliding_puzzle'
    rubiks_cube = 'rubiks_cube'
    watkins_cube = 'watkins_cube'
    known_puzzle_types = [sliding_puzzle, rubiks_cube, watkins_cube]
    n = 'n'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.puzzle_type,
                         type=str,
                         choices=[cls.sliding_puzzle,
                                  cls.rubiks_cube])
        cls.add_argument(parser,
                         field=cls.n,
                         type=int)

    @abstractmethod
    def dimension(self) -> tuple:
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
    def generate_all_puzzles(cls, **kw_args):
        raise NotImplementedError('Implement that when/if possible in concrete Puzzles')

    @classmethod
    def get_move_type(cls):
        assert issubclass(cls.move_type, Move), 'Move type for %s has not been setup properly' % cls.__name__
        return cls.move_type

    @classmethod
    def get_puzzle(cls, nb_shuffles, **kw_args):
        """ return a puzzle shuffled nb_shuffles times randomly from init state
        kw_args allow to construct the puzzle with e.g. puzzle dimension
        """
        puzzle = cls.factory(**kw_args)
        for _ in range(nb_shuffles):
            puzzle = puzzle.apply_random_move()
        return puzzle
    
    @abstractmethod
    def clone(self):
        return

    def get_name(self):
        return '%s[%s]' % (self.__class__.__name__, str(tuple(self.dimension())))

    def is_solvable(self) -> bool:
        raise NotImplementedError('Cannot tell if a %s is solvable' % self.__class__)

    @abstractmethod
    def is_goal(self) -> bool:
        """ True if puzzle in solved state """
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
            nb_shuffles: number of shuffles we do from init state
            nb_sequences: how many such sequences we produce
            min_no_loop: (best effort) making sequences with no loop of length <= min_no_loop
            one_list: results is a list, otherwise a list of lists
        returns:
            list of puzzles
        """
        init = cls(**kw_args)
        training_data = list()
        for _ in range(nb_sequences):
            moves = init.random_moves(nb_shuffles, min_no_loop=min_no_loop)
            puzzles = init.get_puzzle_sequence(moves)
            if one_list:
                training_data += puzzles
            else:
                training_data.append(puzzles)
        return training_data

    @abstractmethod
    def number_of_tiles(self):
        """ How many diff tiles there are in this puzzle """
        return

    @abstractmethod
    def number_of_values(self):
        """ How many diff possible values there are for the tiles """
        return

    @classmethod
    def widget_types(cls):
        from rubiks.puzzle.slidingpuzzle import SlidingPuzzle
        from rubiks.puzzle.rubikscube import RubiksCube
        from rubiks.puzzle.watkinscube import WatkinsCube
        return [SlidingPuzzle, RubiksCube, WatkinsCube]

    @classmethod
    @abstractmethod
    def from_tensor(cls, data, **kw_args):
        """ construct a puzzle from its tensor representation """
        return

    @classmethod
    def factory_key_name(cls):
        return cls.puzzle_type

    @abstractmethod
    def apply(self, move):
        """ return the puzzle resulting from applying move or raise exception if invalid """
        return

    def apply_move(self, move):
        """ alias """
        return self.apply(move)

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
        assert not is_inf(nb_moves)
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

    def possible_puzzles_nb(self):
        """ If known, can overwrite """
        return inf

    def theoretical_moves(self):
        """ return moves in some natural order """
        return

    def theoretical_move(self, move_nb):
        return self.theoretical_moves()[move_nb]

    @abstractmethod
    def possible_moves(self) -> list:
        """ return the set of possible moves from this configuration """
        return

    @classmethod
    def nb_moves(cls):
        """ number of possible moves for the puzzle ... this is not dependent on current game state """
        return

    very_large_nb_shuffle = 1000

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

    def apply_random_moves(self, nb_moves, min_no_loop=None):
        if min_no_loop is None:
            min_no_loop = nb_moves
        if is_inf(nb_moves):
            return self.perfect_shuffle()
        nb_moves = int(nb_moves)
        min_no_loop = int(min_no_loop)
        return self.apply_moves(self.random_moves(nb_moves=nb_moves,
                                                  min_no_loop=min_no_loop))

    @abstractmethod
    def to_tensor(self, one_hot_encoding=False, flatten=True) -> Tensor:
        """ return a torch.Tensor to represent internal state """
        return

    @classmethod
    def optimal_solver_config(cls) -> dict:
        return dict()

    @classmethod
    def custom_goal(cls, goal):
        """ If that one ain't shenanigans :) """
        assert isinstance(goal, cls)

        def is_custom_goal(puzzle):
            return hash(puzzle) in {hash(eq) for eq in goal.get_equivalent()}

        class mod_cls(cls):
            pass

        mod_cls.is_goal = is_custom_goal
        return mod_cls

########################################################################################################################

