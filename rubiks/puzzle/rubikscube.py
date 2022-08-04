########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from enum import Enum
from math import factorial
from numpy.random import randint, permutation
from pandas import DataFrame
from random import choice
from tabulate import tabulate
from torch import ones, zeros, equal, concat
from torch.nn.functional import one_hot
########################################################################################################################
from rubiks.puzzle.puzzle import Move, Puzzle
########################################################################################################################


class Face(Enum):
    F = 'FRONT'
    U = 'UP'
    L = 'LEFT'
    R = 'RIGHT'
    B = 'BACK'
    D = 'DOWN'

########################################################################################################################


class Color(Enum):
    r = 'RED'
    w = 'WHITE'
    g = 'GREEN'
    b = 'BLUE'
    o = 'ORANGE'
    y = 'YELLOW'

########################################################################################################################


rubiks_opposite_color = {Color.r: Color.o,
                         Color.o: Color.r,
                         Color.g: Color.b,
                         Color.b: Color.g,
                         Color.w: Color.y,
                         Color.y: Color.w,
                         }

rubiks_adjacent_colors_clock_wise = dict()
rubiks_adjacent_colors_clock_wise[Color.r] = [Color.w, Color.b, Color.y, Color.g]
rubiks_adjacent_colors_clock_wise[Color.b] = [Color.w, Color.o, Color.y, Color.r]
rubiks_adjacent_colors_clock_wise[Color.o] = [Color.w, Color.g, Color.y, Color.b]
rubiks_adjacent_colors_clock_wise[Color.g] = [Color.w, Color.r, Color.y, Color.o]
rubiks_adjacent_colors_clock_wise[Color.y] = [Color.r, Color.b, Color.o, Color.g]
rubiks_adjacent_colors_clock_wise[Color.w] = [Color.r, Color.g, Color.o, Color.b]

for _ in Color:
    assert rubiks_opposite_color[_] not in rubiks_adjacent_colors_clock_wise[_]
    assert _ not in rubiks_adjacent_colors_clock_wise[_]

########################################################################################################################


rubiks_to_int_map = {Color.r: 1,
                     Color.w: 2,
                     Color.g: 3,
                     Color.b: 4,
                     Color.o: 5,
                     Color.y: 6,
                     Face.F: 1,
                     Face.U: 2,
                     Face.L: 3,
                     Face.R: 4,
                     Face.B: 5,
                     Face.D: 6,
                     }

########################################################################################################################


rubiks_int_to_color_map = {1: Color.r,
                           2: Color.w,
                           3: Color.g,
                           4: Color.b,
                           5: Color.o,
                           6: Color.y,
                           }

########################################################################################################################


rubiks_int_to_face_map = {1: Face.F,
                          2: Face.U,
                          3: Face.L,
                          4: Face.R,
                          5: Face.B,
                          6: Face.D,
                          }

########################################################################################################################


rubiks_color_to_print_color_map = {Color.r: '\033[91m',
                                   Color.w: '\033[90m',
                                   Color.g: '\033[92m',
                                   Color.b: '\033[94m',
                                   Color.o: '\033[38;2;255;165;0m',
                                   Color.y: '\033[93m',
                                   }

########################################################################################################################


def rubiks_to_int(what):
    if isinstance(what, (int, float)):
        return int(what)
    return rubiks_to_int_map[what]

########################################################################################################################


class CubeMove(Move):

    def __init__(self, face: Face, clock_wise: bool = True):
        self.face = face
        self.clock_wise = clock_wise

    def __eq__(self, other):
        return self.face == other.face and self.clock_wise == other.clock_wise

    def __ne__(self, other):
        return self.face != other.face or self.clock_wise != other.clock_wise

    def cost(self):
        return 1

    def __repr__(self):
        return '%s%s' % (self.face.name, '' if self.clock_wise else '\'')

    def opposite(self):
        return CubeMove(self.face, False if self.clock_wise else True)

rubiks_all_moves = list()
for _ in Face:
    rubiks_all_moves.append(CubeMove(_, True))
    rubiks_all_moves.append(CubeMove(_, False))

########################################################################################################################

    
class RubiksCube(Puzzle):
    """ Game of the sliding Puzzle, e.g. the 8-puzzle, 15-puzzle, etc """

    def number_of_tiles(self):
        return 6 * (self.n ** 2)

    def number_of_values(self):
        return 6

    all_faces = {Face.D, Face.L, Face.B, Face.F, Face.R, Face.U}

    tiles = 'tiles'
    init_from_random_goal = 'init_from_random_goal'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.init_from_random_goal,
                         default=False,
                         action=cls.store_true)

    def is_solvable(self) -> bool:
        """ TBD """
        pass

    @classmethod
    def generate_all_puzzles(cls, **kw_args):
        pass

    move_type = CubeMove

    goals_map = dict()
    goals_hashes = set()

    corners_map = dict()

    @classmethod
    def __populate_goals__(cls, n):
        if n in cls.goals_map:
            return
        # there are 6 possible colors for Face.F
        # and then 4 for Face.U, which fixes the rest
        cls.goals_map[n] = list()
        cls.corners_map[n] = list()
        for color in Color:
            tiles = dict()
            tiles[Face.F] = rubiks_to_int(color) * ones(n, n, dtype=int)
            for f, c in zip([Face.U, Face.R, Face.D, Face.L],
                            rubiks_adjacent_colors_clock_wise[color]):
                tiles[f] = rubiks_to_int(c) * ones(n, n, dtype=int)
            tiles[Face.B] = rubiks_to_int(rubiks_opposite_color[color]) * ones(n, n, dtype=int)
            goal = RubiksCube(tiles=tiles)
            cls.goals_map[n].append(goal)
            cls.goals_hashes.add(hash(goal))
        for face in [Face.F, Face.B]:
            color = rubiks_int_to_color_map[rubiks_to_int_map[face]]
            adjacent = [rubiks_to_int_map[c] for c in rubiks_adjacent_colors_clock_wise[color]]
            color = rubiks_to_int_map[color]
            cls.corners_map[n].append((color, adjacent[0], adjacent[1]))
            cls.corners_map[n].append((color, adjacent[1], adjacent[2]))
            cls.corners_map[n].append((color, adjacent[2], adjacent[3]))
            cls.corners_map[n].append((color, adjacent[3], adjacent[-1]))

    def __init__(self, **kw_args):
        from_tiles = self.tiles in kw_args
        if from_tiles and kw_args[self.tiles] is not None:
            self.tiles = kw_args[self.tiles]
            assert all(face in self.tiles for face in self.all_faces)
            self.n = len(self.tiles[Face.U][0])
            assert all(tuple(tiles.shape) == (self.n, self.n) for tiles in self.tiles.values())
        else:
            n = kw_args[self.n]
            self.__populate_goals__(n)
            init_from_random_goal = kw_args.get(self.__class__.init_from_random_goal, False)
            if not init_from_random_goal:
                goal = self.goals_map[n][0]
            else:
                goal = choice(self.goals_map[n])
            self.__init__(tiles={face: tiles.detach().clone() for face, tiles in goal.tiles.items()})

    def __repr__(self):
        tiles = zeros(self.n * 3, self.n * 4, dtype=int)
        tiles[0:self.n, self.n:2 * self.n] = self.tiles[Face.U]
        tiles[self.n:2 * self.n, 0: self.n] = self.tiles[Face.L]
        tiles[self.n:2 * self.n, self.n:2 * self.n] = self.tiles[Face.F]
        tiles[self.n:2 * self.n, 2 * self.n:3 * self.n] = self.tiles[Face.R]
        tiles[self.n:2 * self.n, 3 * self.n:4 * self.n] = self.tiles[Face.B]
        tiles[2 * self.n:3 * self.n, self.n:2 * self.n] = self.tiles[Face.D]
        tiles = DataFrame(tiles)
        tiles = tiles.stack()
        black = '%s%s%s%s' % ('\033[30m',
                              '\u2588',
                              '\u2588',
                              '\033[0m')
        tiles[tiles == 0] = ''
        tiles = tiles.unstack()
        for color in Color:
            col_int = rubiks_to_int_map[color]
            #if color is Color.w:
            #    fill = ''
            #else:
            fill = '%s%s%s%s' % (rubiks_color_to_print_color_map[color],
                                 '\u2588',
                                 '\u2588',
                                 '\033[0m')
            tiles = tiles.stack()
            tiles[tiles == col_int] = fill
            tiles = tiles.unstack()
        tiles = '\n'.join(tabulate(tiles,
                                   headers='keys',
                                   tablefmt='grid',
                                   showindex=False).split('\n')[2:])
        return '\n' + tiles

    def __eq__(self, other):
        for face in Face:
            if not equal(self.tiles[face], other.tiles[face]):
                return False
        return True

    def __hash__(self):
        values = tuple((rubiks_to_int_map[face],
                        hash(tuple(self.tiles[face].flatten().numpy()))) for face in Face)
        return hash(values)

    def dimension(self):
        return (self.n,)*3

    def clone(self):
        return RubiksCube(tiles={face: self.tiles[face].detach().clone() for face in Face})

    def is_goal(self):
        goals = self.goals()
        return hash(self) in self.goals_hashes

    @classmethod
    def construct_puzzle(cls, n, **kw_args):
        return

    def goals(self):
        self.__populate_goals__(self.n)
        return self.goals_map[self.n]

    @staticmethod
    def clock_wise_front(tiles):
        save = tiles[Face.U][-1, :].clone()
        tiles[Face.U][-1, :] = tiles[Face.L][:, -1].flip(0)
        tiles[Face.L][:, -1] = tiles[Face.D][0, :]
        tiles[Face.D][0, :] = tiles[Face.R][:, 0].flip(0)
        tiles[Face.R][:, 0] = save
        tiles[Face.F] = tiles[Face.F].rot90(-1)

    @staticmethod
    def anti_clock_wise_front(tiles):
        save = tiles[Face.U][-1, :].clone().flip(0)
        tiles[Face.U][-1, :] = tiles[Face.R][:, 0]
        tiles[Face.R][:, 0] = tiles[Face.D][0, :].flip(0)
        tiles[Face.D][0, :] = tiles[Face.L][:, -1]
        tiles[Face.L][:, -1] = save
        tiles[Face.F] = tiles[Face.F].rot90()

    @staticmethod
    def clock_wise_right(tiles):
        save = tiles[Face.F][:, -1].clone()
        tiles[Face.F][:, -1] = tiles[Face.D][:, -1]
        tiles[Face.D][:, -1] = tiles[Face.B][:, 0].flip(0)
        tiles[Face.B][:, 0] = tiles[Face.U][:, -1].flip(0)
        tiles[Face.U][:, -1] = save
        tiles[Face.R] = tiles[Face.R].rot90(-1)

    @staticmethod
    def anti_clock_wise_right(tiles):
        save = tiles[Face.F][:, -1].clone()
        tiles[Face.F][:, -1] = tiles[Face.U][:, -1]
        tiles[Face.U][:, -1] = tiles[Face.B][:, 0].flip(0)
        tiles[Face.B][:, 0] = tiles[Face.D][:, -1].flip(0)
        tiles[Face.D][:, -1] = save
        tiles[Face.R] = tiles[Face.R].rot90()

    @staticmethod
    def clock_wise_left(tiles):
        save = tiles[Face.F][:, 0].clone()
        tiles[Face.F][:, 0] = tiles[Face.U][:, 0]
        tiles[Face.U][:, 0] = tiles[Face.B][:, -1].flip(0)
        tiles[Face.B][:, -1] = tiles[Face.D][:, 0].flip(0)
        tiles[Face.D][:, 0] = save
        tiles[Face.L] = tiles[Face.L].rot90(-1)

    @staticmethod
    def anti_clock_wise_left(tiles):
        save = tiles[Face.F][:, 0].clone()
        tiles[Face.F][:, 0] = tiles[Face.D][:, 0]
        tiles[Face.D][:, 0] = tiles[Face.B][:, -1].flip(0)
        tiles[Face.B][:, -1] = tiles[Face.U][:, 0].flip(0)
        tiles[Face.U][:, 0] = save
        tiles[Face.L] = tiles[Face.L].rot90()

    @staticmethod
    def clock_wise_back(tiles):
        save = tiles[Face.U][0, :].clone().flip(0)
        tiles[Face.U][0, :] = tiles[Face.R][:, -1]
        tiles[Face.R][:, -1] = tiles[Face.D][-1, :].flip(0)
        tiles[Face.D][-1, :] = tiles[Face.L][:, 0]
        tiles[Face.L][:, 0] = save
        tiles[Face.B] = tiles[Face.B].rot90(-1)

    @staticmethod
    def anti_clock_wise_back(tiles):
        save = tiles[Face.U][0, :].clone()
        tiles[Face.U][0, :] = tiles[Face.L][:, 0].flip(0)
        tiles[Face.L][:, 0] = tiles[Face.D][-1, :]
        tiles[Face.D][-1, :] = tiles[Face.R][:, -1].flip(0)
        tiles[Face.R][:, -1] = save
        tiles[Face.B] = tiles[Face.B].rot90()

    @staticmethod
    def clock_wise_down(tiles):
        save = tiles[Face.F][-1, :].clone()
        tiles[Face.F][-1, :] = tiles[Face.L][-1, :]
        tiles[Face.L][-1, :] = tiles[Face.B][-1, :]
        tiles[Face.B][-1, :] = tiles[Face.R][-1, :]
        tiles[Face.R][-1, :] = save
        tiles[Face.D] = tiles[Face.D].rot90(-1)

    @staticmethod
    def anti_clock_wise_down(tiles):
        save = tiles[Face.F][-1, :].clone()
        tiles[Face.F][-1, :] = tiles[Face.R][-1, :]
        tiles[Face.R][-1, :] = tiles[Face.B][-1, :]
        tiles[Face.B][-1, :] = tiles[Face.L][-1, :]
        tiles[Face.L][-1, :] = save
        tiles[Face.D] = tiles[Face.D].rot90()

    @staticmethod
    def clock_wise_up(tiles):
        save = tiles[Face.F][0, :].clone()
        tiles[Face.F][0, :] = tiles[Face.R][0, :]
        tiles[Face.R][0, :] = tiles[Face.B][0, :]
        tiles[Face.B][0, :] = tiles[Face.L][0, :]
        tiles[Face.L][0, :] = save
        tiles[Face.U] = tiles[Face.U].rot90(-1)

    @staticmethod
    def anti_clock_wise_up(tiles):
        save = tiles[Face.F][0, :].clone()
        tiles[Face.F][0, :] = tiles[Face.L][0, :]
        tiles[Face.L][0, :] = tiles[Face.B][0, :]
        tiles[Face.B][0, :] = tiles[Face.R][0, :]
        tiles[Face.R][0, :] = save
        tiles[Face.U] = tiles[Face.U].rot90()

    move_functions = dict()
    move_functions[Face.F] = {True: clock_wise_front.__get__(object),
                              False: anti_clock_wise_front.__get__(object)}
    move_functions[Face.R] = {True: clock_wise_right.__get__(object),
                              False: anti_clock_wise_right.__get__(object)}
    move_functions[Face.L] = {True: clock_wise_left.__get__(object),
                              False: anti_clock_wise_left.__get__(object)}
    move_functions[Face.B] = {True: clock_wise_back.__get__(object),
                              False: anti_clock_wise_back.__get__(object)}
    move_functions[Face.D] = {True: clock_wise_down.__get__(object),
                              False: anti_clock_wise_down.__get__(object)}
    move_functions[Face.U] = {True: clock_wise_up.__get__(object),
                              False: anti_clock_wise_up.__get__(object)}

    def apply(self, move: CubeMove):
        puzzle = self.clone()
        self.move_functions[move.face][move.clock_wise](puzzle.tiles)
        return puzzle

    def possible_moves(self):
        return rubiks_all_moves

    @staticmethod
    def __random_move__():
        return CubeMove(face=rubiks_int_to_face_map[randint(1, 7)],
                        clock_wise=randint(0, 2) == 1)

    def random_move(self):
        return self.__random_move__()

    def from_tensor(self):
        raise NotImplementedError('Please implement this ... need to de-one_hot then call init')

    __faces_order__ = [Face.U,
                       Face.R,
                       Face.F,
                       Face.D,
                       Face.L,
                       Face.B]

    def to_tensor(self, one_hot_encoding=False, flatten=True):
        tiles = concat(tuple(self.tiles[f] for f in self.__faces_order__)).to(int) - 1
        if one_hot_encoding:
            tiles = one_hot(tiles, num_classes=6)
        if flatten:
            tiles = tiles.flatten(1)
        return tiles

    @staticmethod
    def opposite(moves):
        return [move.opposite() for move in reversed(moves)]

    def possible_puzzles_nb(self):
        if self.n == 2:
            """ Notice for my purpose each orientation in space is different """
            return factorial(8) * 3**7

    def perfect_shuffle(self):
        if self.n == 2:
            return self.random_2()
        elif self.n == 3:
            return
        else:
            assert False, 'Need to think about that'

    def random_2(self):
        """"""
        assert 2 == self.n
        corners = self.corners_map[self.n]
        targets = [((Face.F, 0, 0), (Face.U, 1, 0), (Face.L, 0, 1)),
                   ((Face.F, 0, 1), (Face.U, 1, 1), (Face.R, 0, 0)),
                   ((Face.F, 1, 0), (Face.D, 0, 0), (Face.L, 1, 1)),
                   ((Face.F, 1, 1), (Face.D, 0, 1), (Face.R, 1, 0)),
                   ((Face.B, 0, 0), (Face.R, 0, 1), (Face.U, 0, 1)),
                   ((Face.B, 0, 1), (Face.L, 0, 0), (Face.U, 0, 0)),
                   ((Face.B, 1, 0), (Face.R, 1, 1), (Face.D, 1, 1)),
                   ((Face.B, 1, 1), (Face.L, 1, 0), (Face.D, 1, 0)),
                   ]
        tiles = dict()
        for target, corner in zip(targets, permutation(corners)):
            for ((face, pos_x, pos_y), color) in zip(target, permutation(corner)):
                if face not in tiles:
                    tiles[face] = zeros(2, 2)
                t = tiles[face]
                t[pos_x, pos_y] = color
        return RubiksCube(tiles=tiles)

    
########################################################################################################################

