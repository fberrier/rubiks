########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from time import time as snap
from torch import tensor
from unittest import TestCase
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.puzzled import Puzzled
from rubiks.utils.utils import ms_format
########################################################################################################################


class TestPuzzled(TestCase):

    def test_puzzled_cannot_construct_without_n(self):
        try:
            logger = Loggable(name='test_puzzled')
            Puzzled(puzzle_type=Puzzle.sliding_puzzle)
            self.assertFalse(True)
        except Exception as error:
            logger.log_info(error)

    def test_puzzled_ok(self):
        try:
            logger = Loggable(name='test_puzzled')
            puzzled = Puzzled(puzzle_type=Puzzle.sliding_puzzle,
                              n=3)
            self.assertEqual(Puzzle.sliding_puzzle, puzzled.get_puzzle_type())
            self.assertEqual('SlidingPuzzle[(3, 3)]', puzzled.puzzle_name())
            self.assertEqual(181440, puzzled.possible_puzzles_nb())
            goal = Puzzled(puzzle_type=Puzzle.sliding_puzzle,
                           tiles=tensor([[1, 2, 3], [4, 5, 6], [7, 8, 0]])).get_goal()
            logger.log_info(goal)
            self.assertEqual(goal, puzzled.get_goal())
            self.assertEqual((3, 3), puzzled.get_puzzle_dimension())
        except Exception as error:
            logger.log_error(error)
            self.assertFalse(True)

    def test_puzzled_repeat(self):
        logger = Loggable(name='test_puzzled_repeat')
        b4 = snap()
        for repeat in range(100000):
            puzzled = Puzzled(puzzle_type=Puzzle.sliding_puzzle, n=10)
        latency = snap() - b4
        logger.log_info({'latency': ms_format(latency)})
        self.assertGreater(10, latency)
        self.assertEqual(Puzzle.sliding_puzzle, puzzled.puzzle_type)
        self.assertEqual('SlidingPuzzle[(10, 10)]', puzzled.puzzle_name())
        self.assertEqual((10, 10), puzzled.get_puzzle_dimension())

    def test_puzzled_from_command_line(self):
        logger = Loggable(name='test_puzzled_from_command_line')
        puzzled = Puzzled.from_command_line("-puzzle_type=sliding_puzzle -n=10 -m=20")
        logger.log_info(puzzled.get_goal())

########################################################################################################################

