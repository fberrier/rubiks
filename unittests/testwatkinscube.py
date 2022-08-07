########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy.random import randint, choice
from unittest import TestCase
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.watkinscube import WatkinsCube
########################################################################################################################


class TestWatkinsCube(TestCase):

    def test_construct(self):
        logger = Loggable(name='test_construct')
        puzzle = WatkinsCube(n=2)
        logger.log_info(puzzle)
        self.assertEqual((2, 2, 2), puzzle.dimension())
        self.assertTrue(puzzle.is_goal())
        puzzle.tiles_goal = puzzle.tiles_goal.whole_cube_up_rotation()
        logger.log_info(puzzle)
        self.assertTrue(puzzle.is_goal())
        puzzle.tiles_goal = puzzle.tiles_goal.whole_cube_right_rotation()
        logger.log_info(puzzle)
        self.assertTrue(puzzle.is_goal())
        puzzle.tiles_goal = puzzle.tiles_goal.whole_cube_front_rotation()
        logger.log_info(puzzle)
        self.assertTrue(puzzle.is_goal())

    def test_one_move(self):
        logger = Loggable(name='test_one_move')
        puzzle = WatkinsCube(n=2).apply_random_moves(1)
        logger.log_info(puzzle)
        self.assertEqual((2, 2, 2), puzzle.dimension())
        self.assertFalse(puzzle.is_goal())
        self.assertTrue(any(puzzle.apply(move).is_goal() for move in puzzle.possible_moves()))

    def test_to_tensor(self):
        logger = Loggable(name='test_to_tensor')
        puzzle = WatkinsCube(n=2).apply_random_moves(3)
        logger.log_info(puzzle)
        self.assertEqual((24, 2), tuple(puzzle.to_tensor().shape))


########################################################################################################################
