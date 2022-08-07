########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from enum import Enum
from itertools import product
from math import factorial
from numpy.random import randint, permutation
from pandas import DataFrame, read_pickle
from random import choice
from tabulate import tabulate
from torch import ones, zeros, concat
from torch.nn.functional import one_hot
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzle import Move, Puzzle
from rubiks.utils.utils import get_file_name, PossibleFileNames, to_pickle
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

    def __init__(self, face: Face, clock_wise: bool = True, whole_cube: bool = False):
        self.face = face
        self.clock_wise = clock_wise
        self.whole_cube = whole_cube

    def __eq__(self, other):
        return self.face == other.face and self.clock_wise == other.clock_wise and self.whole_cube == other.whole_cube

    def __ne__(self, other):
        return self.face != other.face or self.clock_wise != other.clock_wise or self.whole_cube != other.whole_cube

    def cost(self):
        return 1 if not self.whole_cube else 0

    def __repr__(self):
        if not self.whole_cube:
            return '%s%s' % (self.face.name, '' if self.clock_wise else '\'')
        else:
            return 'C%s%s' % (self.face.name, '' if self.clock_wise else '\'')

    def opposite(self):
        return CubeMove(self.face, not self.clock_wise, self.whole_cube)

    @classmethod
    def cleanup_path(cls, path):
        cleanup = True
        while cleanup and len(path) >= 2:
            cleanup = False
            for e, (move_1, move_2) in enumerate(zip(path[:-1], path[1:])):
                if move_1 == move_2.opposite():
                    cleanup = True
                    path = path[:e] + path[e + 2:]
                    break
        return path

rubiks_all_moves = list()
for _ in Face:
    rubiks_all_moves.append(CubeMove(_, True))
    rubiks_all_moves.append(CubeMove(_, False))

########################################################################################################################

    
class RubiksCube(Puzzle):
    """ Rubik's Cube """

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
    goals_hashes = dict()
    corners_map = dict()
    edges_map = dict()
    edges_oriented_distances_to_home = dict()

    def whole_cube_up_rotation(self):
        cube = self.clone()
        cube.tiles[Face.U] = cube.tiles[Face.U].rot90(-1)
        cube.tiles[Face.D] = cube.tiles[Face.D].rot90()
        save = cube.tiles[Face.L].clone()
        cube.tiles[Face.L] = cube.tiles[Face.F]
        cube.tiles[Face.F] = cube.tiles[Face.R]
        cube.tiles[Face.R] = cube.tiles[Face.B]
        cube.tiles[Face.B] = save
        return cube

    def whole_cube_right_rotation(self):
        cube = self.clone()
        cube.tiles[Face.R] = cube.tiles[Face.R].rot90(-1)
        cube.tiles[Face.L] = cube.tiles[Face.L].rot90()
        save = cube.tiles[Face.U].clone().rot90(2)
        cube.tiles[Face.U] = cube.tiles[Face.F]
        cube.tiles[Face.F] = cube.tiles[Face.D]
        cube.tiles[Face.D] = cube.tiles[Face.B].rot90(2)
        cube.tiles[Face.B] = save
        return cube

    def whole_cube_front_rotation(self):
        cube = self.clone()
        cube.tiles[Face.F] = cube.tiles[Face.F].rot90(-1)
        cube.tiles[Face.B] = cube.tiles[Face.B].rot90()
        save = cube.tiles[Face.U].clone().rot90(-1)
        cube.tiles[Face.U] = cube.tiles[Face.L].rot90(-1)
        cube.tiles[Face.L] = cube.tiles[Face.D].rot90(-1)
        cube.tiles[Face.D] = cube.tiles[Face.R].rot90(-1)
        cube.tiles[Face.R] = save
        return cube

    @classmethod
    def __add_corner_to_map__(cls, corner, n):
        corner = tuple(rubiks_to_int(color) for color in corner)
        cls.corners_map[n].append(corner)

    @classmethod
    def __swap__(cls, what):
        assert isinstance(what, tuple), 'RubiksCube.__swap__ expects tuple, got %s instead' % type(what)
        if 2 == len(what):
            return what[1], what[0]
        elif 6 == len(what):
            return tuple(what[p] for p in [3, 4, 5, 0, 1, 2])
        else:
            assert False

    @classmethod
    def __add_edge_to_map__(cls, edge, position, n):
        cls.edges_map[n][edge] = position
        cls.edges_map[n][cls.__swap__(edge)] = cls.__swap__(position)

    def get_equivalent(self):
        """ generate and return all cubes which are visually indistinguishable from self """
        equivalent = list()
        clone = self.clone()
        for r1 in range(4):
            for r2 in range(4):
                equivalent.append(clone)
                clone = clone.whole_cube_front_rotation()
            clone = clone.whole_cube_up_rotation()
        clone = clone.whole_cube_right_rotation()
        for r2 in range(4):
            equivalent.append(clone)
            clone = clone.whole_cube_front_rotation()
        clone = clone.whole_cube_right_rotation()
        clone = clone.whole_cube_right_rotation()
        for r2 in range(4):
            equivalent.append(clone)
            clone = clone.whole_cube_front_rotation()
        return equivalent

    @classmethod
    def __populate_goals__(cls, n):
        if n in cls.goals_map:
            return
        # there are 6 possible colors for Face.F
        # and then 4 for Face.U, which fixes the rest
        #cls.goals_map[n] = list()
        #cls.goals_hashes[n] = set()
        cls.corners_map[n] = list()
        cls.edges_map[n] = dict()
        cls.edges_oriented_distances_to_home[n] = dict()
        goal = dict()
        for face in cls.all_faces:
            goal[face] = rubiks_to_int(face) * ones(n, n, dtype=int)
        goal = RubiksCube(tiles=goal)
        cls.goals_map[n] = goal.get_equivalent()
        cls.goals_hashes[n] = {hash(g) for g in cls.goals_map[n]}
        """ corners """
        cls.__add_corner_to_map__((Color.r, Color.w, Color.g), n)
        cls.__add_corner_to_map__((Color.r, Color.b, Color.w), n)
        cls.__add_corner_to_map__((Color.r, Color.g, Color.y), n)
        cls.__add_corner_to_map__((Color.r, Color.y, Color.b), n)
        cls.__add_corner_to_map__((Color.o, Color.y, Color.g), n)
        cls.__add_corner_to_map__((Color.o, Color.b, Color.y), n)
        cls.__add_corner_to_map__((Color.o, Color.w, Color.b), n)
        cls.__add_corner_to_map__((Color.o, Color.g, Color.w), n)
        if n == 3:
            """ edges """
            cls.__add_edge_to_map__((Color.r, Color.w), (Face.F, 0, 1, Face.U, 2, 1), n)
            cls.__add_edge_to_map__((Color.r, Color.b), (Face.F, 1, 2, Face.R, 1, 0), n)
            cls.__add_edge_to_map__((Color.r, Color.y), (Face.F, 2, 1, Face.D, 0, 1), n)
            cls.__add_edge_to_map__((Color.r, Color.g), (Face.F, 1, 0, Face.L, 1, 2), n)
            cls.__add_edge_to_map__((Color.w, Color.g), (Face.U, 1, 0, Face.L, 0, 1), n)
            cls.__add_edge_to_map__((Color.w, Color.b), (Face.U, 1, 2, Face.R, 0, 1), n)
            cls.__add_edge_to_map__((Color.y, Color.g), (Face.D, 1, 0, Face.L, 2, 1), n)
            cls.__add_edge_to_map__((Color.y, Color.b), (Face.D, 1, 2, Face.R, 2, 1), n)
            cls.__add_edge_to_map__((Color.o, Color.w), (Face.B, 0, 1, Face.U, 0, 1), n)
            cls.__add_edge_to_map__((Color.o, Color.b), (Face.B, 1, 0, Face.R, 1, 2), n)
            cls.__add_edge_to_map__((Color.o, Color.y), (Face.B, 2, 1, Face.D, 2, 1), n)
            cls.__add_edge_to_map__((Color.o, Color.g), (Face.B, 1, 2, Face.L, 1, 0), n)
            """ compute all possible distances from one oriented edge to home
                each edge can be at one of 12 places, in 2 possible orientation
                so we need to compute for 12 * 24 distances.
                Let's get going """
            edges = set()
            positions = set()
            for edge, position in cls.edges_map[n].items():
                if edge not in edges and cls.__swap__(edge) not in edges:
                    edges.add(edge)
                positions.add(position)
            assert len(edges) == 12 and len(positions) == 24
            data_base = get_file_name(Puzzle.rubiks_cube,
                                      dimension=(n, n, n),
                                      file_type=PossibleFileNames.utils,
                                      name='edges_oriented_distances_to_home')
            load_ok = False
            log_info = Loggable(name='edges_oriented_distances_to_home').log_info
            try:
                cls.edges_oriented_distances_to_home[n] = read_pickle(data_base)
                log_info('Loaded ', data_base)
                load_ok = True
            except FileNotFoundError:
                pass
            if load_ok:
                return
            for edge in edges:
                cls.edges_oriented_distances_to_home[n][edge] = dict()
                for position in positions:
                    cls.edges_oriented_distances_to_home[n][edge][position] = \
                        cls.compute_edge_orientated_distance_to_home(edge, position, n)
            to_pickle(cls.edges_oriented_distances_to_home[n], data_base)
            message = 'Saved %d data points to %s' % (len(cls.edges_oriented_distances_to_home[n]),
                                                      data_base)
            log_info(message)

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
        self.check_consistency()

    def check_consistency(self):
        tiles = self.to_tensor()
        for c in range(1, 7):
            assert sum(sum(tiles == c - 1)).item() == self.n ** 2, 'badly formed puzzle \n%s' % self

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
        return hash(other) in {hash(eq) for eq in self.get_equivalent()}

    def __hash__(self):
        values = tuple((rubiks_to_int_map[face],
                        hash(tuple(self.tiles[face].flatten().numpy()))) for face in Face)
        return hash(values)

    def dimension(self):
        return (self.n,)*3

    def clone(self):
        return self.__class__(tiles={face: self.tiles[face].detach().clone() for face in Face})

    def is_goal(self):
        goals = self.goals()
        return hash(self) in self.goals_hashes[self.n]

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
        if not move.whole_cube:
            self.move_functions[move.face][move.clock_wise](puzzle.tiles)
            return puzzle
        if move.face == Face.U:
            if move.clock_wise:
                return self.whole_cube_up_rotation()
            else:
                cube = self.whole_cube_up_rotation()
                cube = cube.whole_cube_up_rotation()
                return cube.whole_cube_up_rotation()
        elif move.face == Face.R:
            if move.clock_wise:
                return self.whole_cube_right_rotation()
            else:
                cube = self.whole_cube_right_rotation()
                cube = cube.whole_cube_right_rotation()
                return cube.whole_cube_right_rotation()
        if move.face == Face.F:
            if move.clock_wise:
                return self.whole_cube_front_rotation()
            else:
                cube = self.whole_cube_front_rotation()
                cube = cube.whole_cube_front_rotation()
                return cube.whole_cube_front_rotation()
        raise NotImplementedError

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
                   ((Face.F, 0, 1), (Face.R, 0, 0), (Face.U, 1, 1)),
                   ((Face.F, 1, 0), (Face.L, 1, 1), (Face.D, 0, 0)),
                   ((Face.F, 1, 1), (Face.D, 0, 1), (Face.R, 1, 0)),
                   ((Face.B, 1, 1), (Face.D, 1, 0), (Face.L, 1, 0)),
                   ((Face.B, 1, 0), (Face.R, 1, 1), (Face.D, 1, 1)),
                   ((Face.B, 0, 0), (Face.U, 0, 1), (Face.R, 0, 1)),
                   ((Face.B, 0, 1), (Face.L, 0, 0), (Face.U, 0, 0)),
                   ]
        tiles = dict()
        p_corners = permutation(corners)
        def permutate(nb, corner):
            corner = list(corner)
            corner.extend(corner[:2])
            r = randint(0, 3)
            corner = corner[r: r+3]
            return corner, r

        total_r = 0
        for nb, (target, corner) in enumerate(zip(targets, p_corners)):
            corner, r = permutate(nb, corner)
            total_r += r
            total_r %= 3
            if nb == 7:
                while total_r != 0:
                    corner, r = permutate(nb, corner)
                    total_r += r
                    total_r %= 3
            for ((face, pos_x, pos_y), color) in zip(target, corner):
                if face not in tiles:
                    tiles[face] = zeros(2, 2)
                t = tiles[face]
                t[pos_x, pos_y] = color
        return RubiksCube(tiles=tiles)

    def edges_orientated_parity(self):
        assert self.n == 3
        positions_checked = set()
        total_parity = 0
        for edge_position in self.edges_map[self.n].values():
            if edge_position in positions_checked:
                continue
            positions_checked.add(edge_position)
            positions_checked.add(self.__swap__(edge_position))
            edge = (self.tiles[edge_position[0]][edge_position[1]][edge_position[2]],
                    self.tiles[edge_position[3]][edge_position[4]][edge_position[5]])
            edge = tuple(rubiks_int_to_color_map[value.item()] for value in edge)
            if edge not in self.edges_oriented_distances_to_home[self.n]:
                edge = self.__swap__(edge)
                total_parity += 1
                assert edge in self.edges_oriented_distances_to_home[self.n], 'WTF?'
            home_position = self.edges_map[self.n][edge]
            total_parity += self.edges_oriented_distances_to_home[self.n][edge][home_position]
        return total_parity % 2

    @classmethod
    def compute_edge_orientated_distance_to_home(cls, edge, position, n):
        from rubiks.solvers.bfssolver import BFSSolver
        target = cls.edges_map[n][edge]
        """ make up a Cube which has say 0 everywhere but the edge we need to solve for """
        cube = RubiksCube(n=n)
        target_cube = RubiksCube(n=n)
        for face in cube.tiles.keys():
            cube.tiles[face] = zeros(n, n)
            target_cube.tiles[face] = zeros(n, n)
        cube.tiles[position[0]][position[1]][position[2]] = rubiks_to_int(edge[0])
        cube.tiles[position[3]][position[4]][position[5]] = rubiks_to_int(edge[1])
        target_cube.tiles[target[0]][target[1]][target[2]] = rubiks_to_int(edge[0])
        target_cube.tiles[target[3]][target[4]][target[5]] = rubiks_to_int(edge[1])
        """ make up goal to be that the edge is in the right position """
        cube = RubiksCube.custom_goal(target_cube)(tiles=cube.tiles)
        """ Finally we run a BFS, which we know at most is 4 so that's manageable """
        solver = BFSSolver(puzzle_type=Puzzle.rubiks_cube, n=n, time_out=60)
        solution = solver.solve(cube)
        assert solution.success and 0 <= solution.cost <= 4,\
            'edge = %s, position = %s, target = %s, solution = %s' % (edge, position, target_cube, solution)
        Loggable(name='compute_edge_orientated_distance_to_home').log_info(edge, position, target, ' -> ', solution.cost)
        return solution.cost

    def swap_edge(self, edge):
        save = self.tiles[edge[0]][edge[1]][edge[2]].item()
        self.tiles[edge[0]][edge[1]][edge[2]] = self.tiles[edge[3]][edge[4]][edge[5]]
        self.tiles[edge[3]][edge[4]][edge[5]] = save

    @staticmethod
    def whole_cube_moves_finder(cube_1, cube_2):
        assert cube_1.n == cube_2.n
        if not cube_1 == cube_2:
            raise ValueError
        if hash(cube_1) == hash(cube_2):
            return list()
        ok_moves = list()
        for face in [Face.F, Face.R, Face.U]:
            for clock_wise in [True, False]:
                ok_moves.append(CubeMove(face=face, whole_cube=True, clock_wise=clock_wise))
        """ Can only be 2 away """
        for move_1 in ok_moves:
            if hash(cube_1.apply(move_1)) == hash(cube_2):
                return [move_1]
        for move_1, move_2 in product(ok_moves, ok_moves):
            if hash(cube_1.apply(move_1).apply(move_2)) == hash(cube_2):
                return [move_1, move_2]
        for move_1, move_2, move_3 in product(ok_moves, ok_moves, ok_moves):
            if hash(cube_1.apply(move_1).apply(move_2).apply(move_3)) == hash(cube_2):
                return [move_1, move_2, move_3]
        for move_1, move_2, move_3, move_4 in product(ok_moves, ok_moves, ok_moves, ok_moves):
            if hash(cube_1.apply(move_1).apply(move_2).apply(move_3).apply(move_4)) == hash(cube_2):
                return [move_1, move_2, move_3, move_4]
        raise RuntimeError
    
########################################################################################################################

