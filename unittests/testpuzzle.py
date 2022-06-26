########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from time import time as snap
from torch import tensor, equal
from unittest import TestCase
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.slidingpuzzle import SlidingPuzzle
from rubiks.utils.utils import ms_format
########################################################################################################################


class TestPuzzle(TestCase):

    def test_puzzle_is_abstract(self):
        try:
            Puzzle(puzzle_type=Puzzle.sliding_puzzle, n=3)
            self.assertFalse(True)
        except TypeError as error:
            pass

    def test_sliding_puzzle(self):
        logger = Loggable(name='test_sliding_puzzle')
        try:
            sliding_puzzle = SlidingPuzzle(n=3)
            self.assertEqual(3, sliding_puzzle.n)
            self.assertEqual(3, sliding_puzzle.m)
            self.assertTrue(equal(sliding_puzzle.tiles,
                                  tensor([[1, 2, 3], [4, 5, 6], [7, 8, 0]])))
            self.assertEqual(sliding_puzzle.empty, (2, 2))
            self.assertTrue(sliding_puzzle.is_goal())
            logger.log_info(sliding_puzzle)
        except TypeError as error:
            logger.log_error(error)
            self.assertFalse(True)
        except Exception as error:
            logger.log_error(error)
            self.assertFalse(True)
        try:
            sliding_puzzle = SlidingPuzzle(n=5, m=10)
            self.assertEqual(5, sliding_puzzle.n)
            self.assertEqual(10, sliding_puzzle.m)
            logger.log_info(sliding_puzzle)
        except TypeError as error:
            logger.log_error(error)
            self.assertFalse(True)
        except Exception as error:
            logger.log_error(error)
            self.assertFalse(True)
        try:
            tiles = tensor([[1, 2, 3, 4], [5, 0, 7, 6]])
            sliding_puzzle = SlidingPuzzle(tiles=tiles)
            self.assertTrue(equal(tiles, sliding_puzzle.tiles))
            self.assertEqual(2, sliding_puzzle.n)
            self.assertEqual(4, sliding_puzzle.m)
            self.assertEqual((1, 1), sliding_puzzle.empty)
            self.assertFalse(sliding_puzzle.is_goal())
            self.assertTrue(sliding_puzzle.goal().is_goal())
            logger.log_info(sliding_puzzle)
            self.assertEqual('SlidingPuzzle[(2, 4)]',
                             sliding_puzzle.get_name())
            logger.log_info(sliding_puzzle.get_name())
        except Exception as error:
            logger.log_error(error)
            self.assertFalse(True)
        try:
            tiles = tensor([[1, 2, 3, 4], [5, 0, 7, 6]])
            sliding_puzzle = SlidingPuzzle(tiles=tiles,
                                           empty=(1, 1))
            self.assertTrue(equal(tiles, sliding_puzzle.tiles))
            self.assertEqual(2, sliding_puzzle.n)
            self.assertEqual(4, sliding_puzzle.m)
            self.assertEqual((1, 1), sliding_puzzle.empty)
            logger.log_info(sliding_puzzle)
        except Exception as error:
            logger.log_error(error)
            self.assertFalse(True)
        self.assertTrue(SlidingPuzzle(n=10).is_goal())

    def test_factory(self):
        logger = Loggable(name='test_factory')
        sliding_puzzle = Puzzle.factory(puzzle_type=Puzzle.sliding_puzzle,
                                        n=3)
        logger.log_info(sliding_puzzle)
        self.assertEqual((3, 3), sliding_puzzle.dimension())

    def test_factory_3_6(self):
        logger = Loggable(name='test_factory_3_6')
        sliding_puzzle = Puzzle.factory(puzzle_type=Puzzle.sliding_puzzle,
                                        n=3,
                                        m=6)
        logger.log_info(sliding_puzzle)
        self.assertEqual((3, 6), sliding_puzzle.dimension())
        logger.log_info(sliding_puzzle.get_config())
        sliding_puzzle_2 = Puzzle.factory(**sliding_puzzle.get_config())
        logger.log_info(sliding_puzzle_2)
        self.assertEqual((3, 6), sliding_puzzle_2.dimension())

    def test_factory_repeat(self):
        logger = Loggable(name='test_factory_repeat')
        b4 = snap()
        for repeat in range(100000):
            Puzzle.factory(puzzle_type=Puzzle.sliding_puzzle, n=10)
        latency = snap() - b4
        logger.log_info({'latency': ms_format(latency)})
        self.assertGreater(10, latency)

    def test_generate_all_sliding_puzzles_2_2(self):
        logger = Loggable(name='test_generate_all_sliding_puzzles_2_2')
        all_puzzles = Puzzle.factory_type(puzzle_type=Puzzle.sliding_puzzle,
                                          n=2).generate_all_puzzles(n=2)
        count = 0
        for puzzle in all_puzzles:
            logger.log_info(puzzle)
            count += 1
        self.assertEqual(puzzle.possible_puzzles_nb(), count)
        logger.log_info('Generated ', count, ' puzzles')


########################################################################################################################

