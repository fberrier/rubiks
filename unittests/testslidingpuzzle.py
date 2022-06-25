########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.puzzle.slidingpuzzle import SlidingPuzzle
########################################################################################################################


class TestSlidingPuzzle(TestCase):

    def test_perfect_shuffle(self):
        puzzle = SlidingPuzzle(n=4, m=4)
        self.assertEqual(puzzle.signature(), puzzle.goal_signature())
        for j in range(1000):
            shuffle = puzzle.perfect_shuffle()
            self.assertEqual(shuffle.signature(), shuffle.goal_signature())
            self.assertEqual(0, shuffle.tiles[shuffle.empty])

########################################################################################################################
