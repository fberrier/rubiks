########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.deeplearning.fullyconnected import FullyConnected
########################################################################################################################


class TestDeepLearning(TestCase):

    def test_fully_connected_direct_construct(self):
        fully_connected = FullyConnected(puzzle_type=Puzzle.sliding_puzzle,
                                         n=4)
        self.assertEqual(FullyConnected.default_layers, fully_connected.layers_description)
        self.assertFalse(fully_connected.one_hot_encoding)
        fully_connected_2 = FullyConnected(puzzle_type=Puzzle.sliding_puzzle,
                                           n=4,
                                           layers_description=(2, 3, 4, 5, 6),
                                           one_hot_encoding=True)
        self.assertEqual((2, 3, 4, 5, 6), fully_connected_2.layers_description)
        self.assertTrue(fully_connected_2.one_hot_encoding)

    def test_fully_connected_from_factory(self):
        fully_connected = DeepLearning.factory(network_type=DeepLearning.fully_connected_net,
                                               puzzle_type=Puzzle.sliding_puzzle,
                                               n=4)
        self.assertEqual(FullyConnected.default_layers, fully_connected.layers_description)
        self.assertFalse(fully_connected.one_hot_encoding)
        fully_connected_2 = DeepLearning.factory(network_type=DeepLearning.fully_connected_net,
                                                 puzzle_type=Puzzle.sliding_puzzle,
                                                 n=3,
                                                 m=2,
                                                 layers_description=(2, 3, 4, 5, 6),
                                                 one_hot_encoding=True)
        self.assertEqual((2, 3, 4, 5, 6), fully_connected_2.layers_description)
        self.assertTrue(fully_connected_2.one_hot_encoding)
        self.assertEqual(fully_connected_2.get_puzzle_dimension(), (3, 2))

########################################################################################################################
