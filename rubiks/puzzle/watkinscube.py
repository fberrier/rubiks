########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from itertools import product
from math import ceil
from numpy.random import randint
from torch import concat
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.rubikscube import RubiksCube, CubeMove, Face
from rubiks.utils.utils import pformat, is_inf
########################################################################################################################

class WatkinsCube(Puzzle):
    """ The aim of this puzzle is to go from a scrambled cube to a particular cube
    i.e. not necessarily the canonical goal """

    move_type = CubeMove

    all_faces = RubiksCube.all_faces

    tiles_goal = 'tiles_goal'
    tiles_start = 'tiles_start'

    def __init__(self, **kw_args):
        from_tiles_goal = self.tiles_goal in kw_args
        if from_tiles_goal and kw_args[self.tiles_goal] is not None:
            self.tiles_goal = RubiksCube(tiles=kw_args[self.tiles_goal])
            self.n = self.tiles_goal.n
            kw_args[self.n] = self.n
        else:
            self.n = kw_args[self.n]
            self.tiles_goal = RubiksCube(**kw_args)
        from_tiles_start = self.tiles_start in kw_args
        if from_tiles_start and kw_args[self.tiles_start] is not None:
            self.tiles_start = RubiksCube(tiles=kw_args[self.tiles_start])
            assert self.tiles_start.n == self.tiles_goal.n
        else:
            self.tiles_start = RubiksCube(**kw_args)

    def check_consistency(self):
        tiles = self.to_tensor()
        for c in range(1, 7):
            assert sum(sum(tiles == c - 1)).item() == 2 * (self.n ** 2), 'badly formed puzzle \n%s' % self

    def __repr__(self):
        return pformat({'start': str(self.tiles_start),
                        'goal': str(self.tiles_goal),
                        })

    def __eq__(self, other):
        return self.tiles_start == other.tiles_start and \
               self.tiles_goal == other.tiles_goal

    def __hash__(self):
        return hash((self.__class__.tiles_start,
                     hash(self.tiles_start),
                     self.__class__.tiles_goal,
                     hash(self.tiles_goal),
                     ))

    def dimension(self):
        return (self.n,)*3

    def clone(self):
        return WatkinsCube(tiles_goal={face: self.tiles_goal.tiles[face].detach().clone() for face in Face},
                           tiles_start={face: self.tiles_start.tiles[face].detach().clone() for face in Face},)

    def is_goal(self):
        return self.tiles_goal == self.tiles_start

    def apply(self, move: CubeMove, goal=False):
        puzzle = self.clone()
        if goal:
            puzzle.tiles_goal = puzzle.tiles_goal.apply(move)
        else:
            puzzle.tiles_start = puzzle.tiles_start.apply(move)
        return puzzle

    def apply_random_move(self):
        random_move = self.random_move()
        if random_move is None:
            return self.clone()
        return self.apply(random_move, goal=randint(0, 2) == 0)

    def apply_random_moves(self, nb_moves, min_no_loop=None):
        if min_no_loop is None:
            min_no_loop = nb_moves
        if is_inf(nb_moves):
            return self.perfect_shuffle()
        cube = self.clone()
        if nb_moves <= 0:
            return cube
        nb_moves = int(nb_moves)
        min_no_loop = int(min_no_loop)
        nb_moves_start = ceil(randint(0, nb_moves + 1))
        min_no_loop_start = ceil(min_no_loop * nb_moves_start / nb_moves)
        cube.tiles_start = cube.tiles_start.apply_random_moves(nb_moves_start,
                                                               min_no_loop_start)
        cube.tiles_goal = cube.tiles_goal.apply_random_moves(nb_moves - nb_moves_start,
                                                             min_no_loop - min_no_loop_start)
        return cube

    def possible_moves(self):
        return self.tiles_start.possible_moves()

    def random_move(self):
        return RubiksCube.__random_move__()

    def from_tensor(self):
        raise NotImplementedError('Please implement this ... need to de-one_hot then call init')

    def to_tensor(self, one_hot_encoding=False, flatten=True):
        tiles_start = self.tiles_start.to_tensor(one_hot_encoding=one_hot_encoding, flatten=flatten)
        tiles_goal = self.tiles_goal.to_tensor(one_hot_encoding=one_hot_encoding, flatten=flatten)
        tiles = concat((tiles_start, tiles_goal))
        if flatten:
            tiles = tiles.flatten(1)
        return tiles

    @staticmethod
    def opposite(moves):
        return RubiksCube.opposite(moves)

    def possible_puzzles_nb(self):
        return self.tiles_start.possible_puzzles_nb() ** 2

    def perfect_shuffle(self):
        cube = self.clone()
        cube.tiles_start = cube.tiles_start.perfect_shuffle()
        cube.tiles_goal = cube.tiles_goal.perfect_shuffle()
        return cube

    def number_of_tiles(self):
        return self.tiles_start.number_of_tiles() + self.tiles_goal.number_of_tiles()

    def number_of_values(self):
        return 6

    @classmethod
    def get_training_data(cls,
                          nb_shuffles,
                          nb_sequences,
                          min_no_loop=None,
                          one_list=False,
                          **kw_args):
        """
        modify this so that the shuffles are applied to start and goal alternatively
        """
        if min_no_loop is None or not min_no_loop:
            min_no_loop = nb_shuffles
        init = cls(**kw_args)
        training_data = list()
        nb_start_shuffles = ceil(nb_shuffles / 2)
        nb_goal_shuffles = nb_shuffles - nb_start_shuffles
        assert nb_start_shuffles
        for _ in range(nb_sequences):
            start_moves = init.tiles_start.random_moves(nb_start_shuffles, min_no_loop=min_no_loop)
            goal_moves = init.tiles_goal.random_moves(nb_goal_shuffles, min_no_loop=min_no_loop)
            start_puzzles = init.tiles_start.get_puzzle_sequence(start_moves)
            goal_puzzles = init.tiles_start.get_puzzle_sequence(goal_moves)
            puzzles = list()
            for start_puzzle, goal_puzzle in product(start_puzzles, goal_puzzles):
                puzzle = WatkinsCube(tiles_start=start_puzzle.tiles,
                                     tiles_goal=goal_puzzle.tiles)
                puzzles.append(puzzle)
            if one_list:
                training_data += puzzles
            else:
                training_data.append(puzzles)
        return training_data

########################################################################################################################

