########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.rubikscube import RubiksCube
########################################################################################################################


class TestRubiksCube(TestCase):

    def test_construct(self):
        logger = Loggable(name='test_construct')
        puzzle = RubiksCube(n=3)
        logger.log_info(puzzle)
        self.assertEqual(3, puzzle.dimension())
        self.assertTrue(puzzle.is_goal())

########################################################################################################################
