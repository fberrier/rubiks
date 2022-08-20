########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.puzzle.puzzle import Puzzle
from rubiks.learners.learner import Learner
from rubiks.learners.perfectlearner import PerfectLearner
from rubiks.learners.deeplearner import DeepLearner
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
########################################################################################################################


class TestLearner(TestCase):

    def test_learner_direct_construct_perfect_learner_4_4(self):
        learner = PerfectLearner(puzzle_type=Puzzle.sliding_puzzle,
                                 learning_file_name='test_learner_direct_construct.pkl',
                                 n=4)
        self.assertEqual(learner.nb_cpus, PerfectLearner.default_nb_cpus)
        self.assertEqual(learner.regular_save, PerfectLearner.default_regular_save)
        self.assertEqual(learner.cpu_multiplier, PerfectLearner.default_cpu_multiplier)
        self.assertEqual(learner.get_puzzle_dimension(), (4, 4))

    def test_learner_direct_construct_perfect_learner_4_6(self):
        learner = PerfectLearner(puzzle_type=Puzzle.sliding_puzzle,
                                 learning_file_name='test_learner_direct_construct.pkl',
                                 n=4,
                                 m=6,
                                 cpu_multiplier=45)
        self.assertEqual(learner.nb_cpus, PerfectLearner.default_nb_cpus)
        self.assertEqual(learner.regular_save, PerfectLearner.default_regular_save)
        self.assertEqual(learner.cpu_multiplier, 45)
        self.assertEqual(learner.get_puzzle_dimension(), (4, 6))
        learner_copy = PerfectLearner(**learner.get_config())
        self.assertEqual(learner_copy.nb_cpus, PerfectLearner.default_nb_cpus)
        self.assertEqual(learner_copy.regular_save, PerfectLearner.default_regular_save)
        self.assertEqual(learner_copy.cpu_multiplier, 45)
        self.assertEqual(learner_copy.get_puzzle_dimension(), (4, 6))
        learner_copy_again = PerfectLearner(**{**learner_copy.get_config(),
                                               'cpu_multiplier': 56})
        self.assertEqual(learner_copy_again.nb_cpus, PerfectLearner.default_nb_cpus)
        self.assertEqual(learner_copy_again.regular_save, PerfectLearner.default_regular_save)
        self.assertEqual(learner_copy_again.cpu_multiplier, 56)
        self.assertEqual(learner_copy_again.get_puzzle_dimension(), (4, 6))

    def test_learner_perfect_from_factory(self):
        learner = Learner.factory(learner_type=Learner.perfect_learner,
                                  puzzle_type=Puzzle.sliding_puzzle,
                                  learning_file_name='test_search_strategy_perfect_from_factory.pkl',
                                  n=4)
        self.assertEqual(learner.nb_cpus, PerfectLearner.default_nb_cpus)
        self.assertEqual(learner.regular_save, PerfectLearner.default_regular_save)
        self.assertEqual(learner.cpu_multiplier, PerfectLearner.default_cpu_multiplier)
        self.assertEqual(learner.get_puzzle_dimension(), (4, 4))

    def test_deep_reinforcement_learning_direct_construct(self):
        learner = DeepReinforcementLearner(puzzle_type=Puzzle.sliding_puzzle,
                                           n=4,
                                           m=7,
                                           network_type=DeepLearning.fully_connected_net,
                                           nb_epochs=10,
                                           nb_shuffles=51)
        self.assertEqual(learner.nb_cpus, DeepReinforcementLearner.default_nb_cpus)
        self.assertEqual(learner.update_target_network_frequency,
                         DeepReinforcementLearner.default_update_target_network_frequency)
        self.assertEqual(learner.nb_epochs, 10)
        self.assertEqual(learner.nb_shuffles, 51)
        self.assertEqual(learner.get_puzzle_dimension(), (4, 7))

    def test_deep_learning_from_factory(self):
        learner = Learner.factory(learner_type=Learner.deep_learner,
                                  puzzle_type=Puzzle.sliding_puzzle,
                                  n=7,
                                  m=8,
                                  network_type=DeepLearning.fully_connected_net,
                                  nb_shuffles=10)
        self.assertEqual(learner.nb_cpus, DeepLearner.default_nb_cpus)
        self.assertEqual(learner.get_puzzle_dimension(), (7, 8))

    def test_deep_reinforcement_learning_from_factory(self):
        learner = Learner.factory(learner_type=Learner.deep_reinforcement_learner,
                                  puzzle_type=Puzzle.sliding_puzzle,
                                  n=4,
                                  m=5,
                                  network_type=DeepLearning.fully_connected_net)
        self.assertEqual(learner.nb_cpus, DeepReinforcementLearner.default_nb_cpus)
        self.assertEqual(learner.get_puzzle_dimension(), (4, 5))

########################################################################################################################
