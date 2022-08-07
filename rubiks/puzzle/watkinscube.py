########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from torch import concat
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.rubikscube import RubiksCube, CubeMove, Color, Face
from rubiks.utils.utils import pformat
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
        else:
            self.n = kw_args[self.n]
            self.tiles_goal = RubiksCube(n=self.n)
        from_tiles_start = self.tiles_start in kw_args
        if from_tiles_start and kw_args[self.tiles_start] is not None:
            self.tiles_start = RubiksCube(tiles=kw_args[self.tiles_start])
            assert self.tiles_start.n == self.tiles_goal.n
        else:
            self.tiles_start = RubiksCube(n=self.n)

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

    def apply(self, move: CubeMove):
        puzzle = self.clone()
        puzzle.tiles_start = puzzle.tiles_start.apply(move)
        return puzzle

    def possible_moves(self):
        return self.tiles_start.possible_moves()

    def random_move(self):
        return RubiksCube.random_move()

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
        self.tiles_start = self.tiles_start.perfect_shuffle()
        self.tiles_goal = self.tiles_goal.perfect_shuffle()

    def number_of_tiles(self):
        return self.tiles_start.number_of_tiles() + self.tiles_goal.number_of_tiles()

    def number_of_values(self):
        return 6

    @classmethod
    def get_training_data(cls,
                          nb_shuffles,
                          nb_sequences,
                          min_no_loop=1,
                          one_list=False,
                          **kw_args):
        """
        modify this so that the shuffles are applied to start and goal alternatively
        """
        init = cls(**kw_args)
        training_data = list()
        for _ in range(nb_sequences):
            pass
            """ TBD """
        return training_data

########################################################################################################################

