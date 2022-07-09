########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.core.loggable import Loggable
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

    def test_convolutional_net_from_factory(self):
        logger = Loggable(name='test_convolutional_net_from_factory')
        puzzle_type = Puzzle.sliding_puzzle
        n = 3
        m = 4
        network_type = DeepLearning.convolutional_net
        one_hot_encoding = False
        kernel_size = (3, 3)
        convo_layers_description = (10, 5, 1)
        padding = 1
        kw_args = locals()
        kw_args.pop('self', None)
        puzzle = Puzzle.factory(**kw_args).apply_random_moves(10)
        convo = DeepLearning.factory(**kw_args)
        logger.log_debug(puzzle)
        logger.log_debug(convo.massage_puzzles(puzzle))
        logger.log_debug(convo.evaluate(puzzle))
        logger.log_debug(convo.evaluate(puzzle).shape)
        self.assertEqual((3, 4), tuple(convo.evaluate(puzzle).shape))

    def test_convolutional_net_from_factory_many_puzzles(self):
        logger = Loggable(name='test_convolutional_net_from_factory')
        puzzle_type = Puzzle.sliding_puzzle
        n = 3
        m = 4
        network_type = DeepLearning.convolutional_net
        one_hot_encoding = False
        kernel_size = (3, 3)
        convo_layers_description = (10, 5, 1)
        padding = 1
        kw_args = locals()
        kw_args.pop('self', None)
        puzzles = [Puzzle.factory(**kw_args).apply_random_moves(10) for _ in range(10)]
        convo = DeepLearning.factory(**kw_args)
        logger.log_debug(puzzles)
        logger.log_debug(convo.massage_puzzles(puzzles))
        logger.log_debug(convo.evaluate(puzzles))
        logger.log_debug(convo.evaluate(puzzles).shape)
        self.assertEqual((10, 3, 4), tuple(convo.evaluate(puzzles).shape))

    def test_convolutional_net_with_fully_connected_from_factory(self):
        logger = Loggable(name='test_convolutional_net_with_fully_connected_from_factory')
        puzzle_type = Puzzle.sliding_puzzle
        n = 4
        m = 4
        network_type = DeepLearning.convolutional_net
        one_hot_encoding = False
        kernel_size = (2, 2)
        convo_layers_description = (100, 100, 100)
        convo_layers_description = (100, 10, 1)
        padding = 0
        kw_args = locals()
        kw_args.pop('self', None)
        puzzle = Puzzle.factory(**kw_args).apply_random_moves(10)
        convo = DeepLearning.factory(**kw_args)
        logger.log_debug(puzzle)
        logger.log_debug(convo.massage_puzzles(puzzle))
        logger.log_debug(convo.evaluate(puzzle))
        logger.log_debug(convo.evaluate(puzzle).shape)
        self.assertEqual((), tuple(convo.evaluate(puzzle).shape))

    def test_convolutional_net_with_fully_connected_from_factory_many_puzzles(self):
        logger = Loggable(name='test_convolutional_net_with_fully_connected_from_factory_many_puzzles')
        puzzle_type = Puzzle.sliding_puzzle
        n = 4
        m = 4
        network_type = DeepLearning.convolutional_net
        one_hot_encoding = False
        kernel_size = (2, 2)
        convo_layers_description = (100, 100, 100)
        convo_layers_description = (100, 10, 1)
        padding = 0
        kw_args = locals()
        kw_args.pop('self', None)
        n_samples = 100
        puzzles = [Puzzle.factory(**kw_args).apply_random_moves(10) for _ in range(n_samples)]
        convo = DeepLearning.factory(**kw_args)
        logger.log_debug(puzzles)
        logger.log_debug(convo.massage_puzzles(puzzles))
        logger.log_debug(convo.evaluate(puzzles))
        logger.log_debug(convo.evaluate(puzzles).shape)
        self.assertEqual((n_samples,), tuple(convo.evaluate(puzzles).shape))

########################################################################################################################
