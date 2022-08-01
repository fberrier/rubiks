########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.rubikscube import RubiksCube, CubeMove, Face
########################################################################################################################


class TestRubiksCube(TestCase):

    def test_construct(self):
        logger = Loggable(name='test_construct')
        puzzle = RubiksCube(n=3)
        logger.log_info(puzzle)
        self.assertEqual(3, puzzle.dimension())
        self.assertTrue(puzzle.is_goal())

    def simple_face(self, face):
        logger = Loggable(name='simple_face_%s' % face.value.lower())
        puzzle = RubiksCube(n=3)
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
        self.assertTrue(puzzle.is_goal())
        for _ in range(4):
            puzzle = puzzle.apply(CubeMove(face, False))
        self.assertTrue(puzzle.is_goal())

    def test_move_front(self):
        self.simple_face(Face.F)

    def test_move_right(self):
        self.simple_face(Face.R)

    def test_move_left(self):
        self.simple_face(Face.L)

    def test_move_back(self):
        self.simple_face(Face.B)

    def test_move_down(self):
        self.simple_face(Face.D)

    def test_move_up(self):
        self.simple_face(Face.U)

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
        self.random_moves_and_back(2, 1)
        self.random_moves_and_back(2, 10)
        self.random_moves_and_back(2, 100)
        self.random_moves_and_back(2, 1000)

    def test_random_moves_and_back_3(self):
        self.random_moves_and_back(3, 1)
        self.random_moves_and_back(3, 10)
        self.random_moves_and_back(3, 100)
        self.random_moves_and_back(3, 1000)

    def test_to_kociemba(self):
        logger = Loggable(name='test_to_kociemba')
        kociemba_repr = RubiksCube(n=3).to_kociemba()
        logger.log_info(kociemba_repr)

########################################################################################################################
