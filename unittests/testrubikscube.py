########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy.random import randint, choice
from unittest import TestCase
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.solvers.kociembasolver import KociembaSolver
from rubiks.puzzle.rubikscube import RubiksCube, CubeMove, Face
########################################################################################################################


class TestRubiksCube(TestCase):

    def test_from_kociemba(self):
        logger = Loggable(name='test_from_kociemba')
        solution_string = 'U1 F3 U3 R1 F3 R2 U2 R3 U1'
        solution = KociembaSolver.from_kociemba(solution_string)
        logger.log_info(solution)
        self.assertEqual([CubeMove(Face.U),
                          CubeMove(Face.F, False),
                          CubeMove(Face.U, False),
                          CubeMove(Face.R),
                          CubeMove(Face.F, False),
                          CubeMove(Face.R),
                          CubeMove(Face.R),
                          CubeMove(Face.U),
                          CubeMove(Face.U),
                          CubeMove(Face.R, False),
                          CubeMove(Face.U)],
                         solution)

    def test_construct(self):
        logger = Loggable(name='test_construct')
        puzzle = RubiksCube(n=3)
        logger.log_info(puzzle)
        self.assertEqual((3, 3, 3), puzzle.dimension())
        self.assertTrue(puzzle.is_goal())

    def simple_face(self, face, n):
        logger = Loggable(name='simple_face_%s' % face.value.lower())
        puzzle = RubiksCube(n=n)
        logger.log_info(puzzle)
        puzzle = puzzle.apply(CubeMove(face))
        logger.log_info(puzzle)
        puzzle = puzzle.apply(CubeMove(face, False))
        logger.log_info(puzzle)
        puzzle = puzzle.apply(CubeMove(face, False))
        logger.log_info(puzzle)
        puzzle = puzzle.apply(CubeMove(face))
        logger.log_info(puzzle)
        self.assertTrue(puzzle.is_goal())
        for _ in range(4):
            puzzle = puzzle.apply(CubeMove(face))
            logger.log_info(puzzle)
        self.assertTrue(puzzle.is_goal())
        for _ in range(4):
            puzzle = puzzle.apply(CubeMove(face, False))
            logger.log_info(puzzle)
        self.assertTrue(puzzle.is_goal())

    n_max = 10

    def test_move_front(self):
        for n in range(1, self.n_max):
            self.simple_face(Face.F, n)

    def test_move_right(self):
        for n in range(1, self.n_max):
            self.simple_face(Face.R, n)

    def test_move_left(self):
        for n in range(1, self.n_max):
            self.simple_face(Face.L, n)

    def test_move_back(self):
        for n in range(1, self.n_max):
            self.simple_face(Face.B, n)

    def test_move_down(self):
        for n in range(1, self.n_max):
            self.simple_face(Face.D, n)

    def test_move_up(self):
        for n in range(1, self.n_max):
            self.simple_face(Face.U, n)

    def random_moves_and_back(self, size, nb_shuffles):
        logger = Loggable(name='test_random_moves_and_back_%d_%d' % (size, nb_shuffles))
        moves = list()
        for _ in range(nb_shuffles):
            move = RubiksCube.__random_move__()
            moves.append(move)
        puzzle = RubiksCube(n=size)
        self.assertTrue(puzzle.is_goal())
        logger.log_info(puzzle)
        puzzle = puzzle.apply_moves(moves)
        logger.log_info(puzzle)
        puzzle = puzzle.apply_moves(puzzle.opposite(moves))
        logger.log_info(puzzle)
        self.assertTrue(puzzle.is_goal())

    def test_random_moves_and_back_2(self):
        for size in [1, 10, 100, 1000]:
            self.random_moves_and_back(2, size)

    def test_random_moves_and_back_3(self):
        for size in [1, 10, 100, 1000]:
            self.random_moves_and_back(3, size)

    def test_to_kociemba(self):
        logger = Loggable(name='test_to_kociemba')
        kociemba_repr = KociembaSolver.to_kociemba(RubiksCube(n=3))
        logger.log_info(kociemba_repr)
        self.assertEqual('UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB',
                         kociemba_repr)

    def test_edge_orientation_parity(self):
        number_of_times = 100
        for _ in range(number_of_times):
            cube = RubiksCube(n=3).apply_random_moves(randint(0, number_of_times))
            parity = cube.edges_orientated_parity()
            assert 0 == parity
        edges = list(RubiksCube(n=3).edges_map[3].values())
        for _ in range(number_of_times):
            cube = RubiksCube(n=3).apply_random_moves(randint(0, number_of_times))
            cube.swap_edge(edges[choice(len(edges))])
            parity = cube.edges_orientated_parity()
            assert 1 == parity

    def test_custom_goal(self):
        number_of_times = 1
        for _ in range(number_of_times):
            cube = RubiksCube(n=3).apply_random_moves(number_of_times)
            custom_goal = cube.apply_random_moves(1)
            cumstom_rubiks = RubiksCube.custom_goal(custom_goal)
            cube = cumstom_rubiks(tiles=cube.tiles)
            possible_moves = cube.possible_moves()
            self.assertTrue(any(cube.apply(move).is_goal() for move in possible_moves))

    def test_equivalent(self):
        cube = RubiksCube(n=2)
        goals = {hash(goal) for goal in cube.goals()}
        equivalent = {hash(eq) for eq in cube.get_equivalent()}
        self.assertEqual(goals, equivalent)

    def test_equivalent_2(self):
        number_of_times = 100
        for _ in range(number_of_times):
            cube = RubiksCube(n=2).apply_random_moves(number_of_times)
            self.assertEqual(24, len({hash(eq) for eq in cube.get_equivalent()}))

    def test_goals(self):
        self.assertEqual(24, len(RubiksCube(n=2).goals()))
        self.assertEqual(24, len(RubiksCube.goals_hashes[2]))

    def test_equivalent_1_1_1(self):
        cube = RubiksCube(n=1)
        self.assertEqual(24, len(cube.get_equivalent()))
        self.assertEqual(24, len({hash(eq) for eq in cube.get_equivalent()}))

    def test_cleanup_path(self):
        moves = [CubeMove(Face.U),
                 CubeMove(Face.R),
                 CubeMove(Face.L),
                 CubeMove(Face.R),
                 CubeMove(Face.L, False),
                 ]
        self.assertEqual(len(moves), len(CubeMove.cleanup_path(moves)))
        moves = [CubeMove(Face.U),
                 CubeMove(Face.R),
                 CubeMove(Face.L),
                 CubeMove(Face.L, False),
                 ]
        self.assertEqual(2, len(CubeMove.cleanup_path(moves)))
        moves = [CubeMove(Face.U),
                 CubeMove(Face.R),
                 CubeMove(Face.R, False),
                 CubeMove(Face.L, False),
                 ]
        self.assertEqual(2, len(CubeMove.cleanup_path(moves)))
        moves = [CubeMove(Face.U),
                 CubeMove(Face.R),
                 CubeMove(Face.R, False),
                 CubeMove(Face.U, False),
                 ]
        self.assertEqual(0, len(CubeMove.cleanup_path(moves)))

########################################################################################################################
