########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy.random import randint
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

    def random_moves_and_back(self, size, nb_shuffles):
        logger = Loggable(name='test_random_moves_and_back_%d_%d' % (size, nb_shuffles))
        moves = list()
        for _ in range(nb_shuffles):
            face_rand = randint(0, 3)
            move = CubeMove(Face.F if face_rand == 0 else Face.R if face_rand == 1 else Face.L,
                            True if randint(0, 2) == 0 else False)
            moves.append(move)
        puzzle = RubiksCube(n=size)
        self.assertTrue(puzzle.is_goal())
        logger.log_info(puzzle)
        puzzle = puzzle.apply_moves(moves)
        logger.log_info(puzzle)
        puzzle = puzzle.apply_moves(puzzle.opposite(moves))
        logger.log_info(puzzle)
        self.assertTrue(puzzle.is_goal())

    def test_random_moves_and_back_3_100(self):
        self.random_moves_and_back(3, 100)


########################################################################################################################
