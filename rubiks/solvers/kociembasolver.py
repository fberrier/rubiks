########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.puzzle.rubikscube import RubiksCube, Face, rubiks_int_to_face_map, CubeMove
from rubiks.puzzle.watkinscube import WatkinsCube
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class KociembaSolver(Solver):
    """ Rubik's solver from Kociemba """

    __import_done__ = False
    __kociemba_solve__ = None
    __hkociemba_solve__ = None

    @staticmethod
    def to_kociemba(puzzle):
        assert puzzle.n in {2, 3}, 'Kociemba only works for 3*3*3 & 2*2*2 Rubik\'s'
        faces_order = [Face.U,
                       Face.R,
                       Face.F,
                       Face.D,
                       Face.L,
                       Face.B]
        tiles = ''
        for face in faces_order:
            tiles += ''.join([rubiks_int_to_face_map[int(tile)].name for tile in puzzle.tiles[face].flatten()])
        return tiles

    @staticmethod
    def from_kociemba(solution):
        solution = solution.split(' ')
        moves = list()
        for move in solution:
            if move.strip() == '':
                continue
            double = move.find('2') >= 0
            triple = move.find('3') >= 0
            anti = move.find('\'') >= 0
            if triple and not anti:
                anti = True
            face = Face[move[0].upper()]
            move = CubeMove(face=face, clock_wise=not anti)
            moves.append(move)
            if double:
                moves.append(move)
            moves = CubeMove.cleanup_path(moves)
        return moves

    def known_to_be_optimal(self):
        return self.get_puzzle_dimension()[0] == 2# and self.get_puzzle_type() is RubiksCube

    def __do_import__(self):
        if self.__class__.__import_done__:
            return
        try:
            from kociemba import solve as kociemba_solve
            self.__class__.__kociemba_solve__ = kociemba_solve
        except ModuleNotFoundError:
            pass
        try:
            from rubiks.thirdparties.hkociemba.solver import solve as hkociemba_solve
            self.__class__.__hkociemba_solve__ = hkociemba_solve
        except ModuleNotFoundError:
            pass
        self.__class__.__import_done__ = True

    def kociemba_solve(self, cube_string):
        self.__do_import__()
        return self.__class__.__kociemba_solve__(cube_string)

    def hkociemba_solve(self, cube_string):
        self.__do_import__()
        return self.__class__.__hkociemba_solve__(cube_string)

    def solve_impl_rubiks(self, puzzle, **kw_args) -> Solution:
        assert isinstance(puzzle, RubiksCube) and puzzle.dimension() in {(3, 3, 3), (2, 2, 2)}
        cube_string = self.to_kociemba(puzzle)
        try:
            if puzzle.is_goal():
                solution_string = ''
                moves = list()
            else:
                solution_string = self.kociemba_solve(cube_string) if puzzle.dimension()[0] == 3 else \
                    self.hkociemba_solve(cube_string)
                moves = self.from_kociemba(solution_string)
            solution = Solution(cost=len(moves),
                                path=moves,
                                expanded_nodes=float('nan'),
                                puzzle=puzzle,
                                solver_name=self.get_name(),
                                cube_string=cube_string,
                                solution_string=solution_string,
                                )
        except Exception as error:
            self.log_error(error)
            raise RuntimeError('%s[%s]' % (error, cube_string))
        return solution

    def solve_impl(self, puzzle, **kw_args) -> Solution:
        assert isinstance(puzzle, (RubiksCube, WatkinsCube))
        if isinstance(puzzle, RubiksCube):
            return self.solve_impl_rubiks(puzzle, **kw_args)
        elif isinstance(puzzle, WatkinsCube):
            from_start = self.solve_impl_rubiks(puzzle.tiles_start, **kw_args)
            from_goal = self.solve_impl_rubiks(puzzle.tiles_goal, **kw_args)
            intermediary_1 = from_start.apply(puzzle.tiles_start)
            intermediary_2 = from_goal.apply(puzzle.tiles_goal)
            rotation_moves = list()
            if hash(intermediary_1) != hash(intermediary_2):
                """ Most likely case, well 23/24 ... we need to rotate cube """
                rotation_moves = RubiksCube.whole_cube_moves_finder(intermediary_1, intermediary_2)
            path = from_start.path + rotation_moves + list(move.opposite() for move in reversed(from_goal.path))
            path = CubeMove.cleanup_path(path)
            cost = sum(move.cost() for move in path) # notice that the full rotation I count as 0
            solution = Solution(cost=cost,
                                path=path,
                                expanded_nodes=float('nan'),
                                puzzle=puzzle,
                                solver_name=self.get_name(),
                                cube_string=[from_start.additional_info['cube_string'],
                                             from_goal.additional_info['cube_string']],
                                solution_string=[from_start.additional_info['solution_string'],
                                                 ''.join(reversed(from_goal.additional_info['solution_string']))],
                                )
            return solution
        else:
            raise NotImplementedError('Kociemba solver does not support %s' % type(puzzle))

########################################################################################################################
