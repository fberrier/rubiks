########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy.random import permutation
from unittest import TestCase
########################################################################################################################
from rubiks.utils.utils import out_of_order, bubble_sort_swaps_count
########################################################################################################################


class TestArgumentParser(TestCase):

    def test_out_of_order(self):
        some_permutation = [1, 2, 3]
        self.assertEqual(0, out_of_order(some_permutation))
        some_permutation = [3, 2, 1]
        self.assertEqual(3, out_of_order(some_permutation))
        some_permutation = [1, 3, 2]
        self.assertEqual(1, out_of_order(some_permutation))
        some_permutation = [1, 2, 3, 4, 5, 6]
        self.assertEqual(0, out_of_order(some_permutation))
        some_permutation = [1, 3, 2, 4, 6, 5]
        self.assertEqual(2, out_of_order(some_permutation))

    def test_out_of_order_and_bubble_sort_swaps_count(self):
        """ Testing my theory """
        for max_val in range(1, 100):
            some_permutation = permutation(range(max_val))
            self.assertEqual(out_of_order(some_permutation),
                             bubble_sort_swaps_count(some_permutation))

########################################################################################################################
