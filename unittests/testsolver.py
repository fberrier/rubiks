########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
from math import inf
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.perfectlearner import PerfectLearner
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.utils.utils import get_model_file_name, remove_file
########################################################################################################################


class TestSolver(TestCase):

    def test_solver(self):
        # just instantiate an a* solver
        logger = Loggable(name='test_solver')
        solver = Solver.factory(solver_type=Solver.astar,
                                puzzle_type=Puzzle.sliding_puzzle,
                                n=2,
                                m=3)
        logger.log_info(solver.get_config())
        self.assertEqual(solver.puzzle_type, Puzzle.sliding_puzzle)
        self.assertEqual(solver.solver_type, Solver.astar)
        self.assertEqual(solver.get_puzzle_dimension(), (2, 3))

    def test_a_star_perfect_solver(self):
        logger = Loggable(name='test_a_star_perfect_solver')
        puzzle_type = Puzzle.sliding_puzzle
        dimension = (2, 2)
        # we learn first
        model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='test_a_star_perfect_solver')
        remove_file(model_file_name)
        learner = PerfectLearner(puzzle_type=Puzzle.sliding_puzzle,
                                 learning_file_name=model_file_name,
                                 solver_type=Solver.astar,
                                 heuristic_type=Heuristic.manhattan,
                                 time_out=2,
                                 n=dimension[0],
                                 m=dimension[1])
        logger.log_info(learner.get_config())
        learner.learn()
        # Then we use this learning to solve
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type=Heuristic.perfect,
                                model_file_name=model_file_name,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1])
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(10)
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        logger.log_info(solution)
        remove_file(model_file_name)

    def test_solver_a_star_manhattan(self):
        logger = Loggable(name='test_solver_a_star_manhattan')
        dimension = (3, 4)
        puzzle_type = Puzzle.sliding_puzzle
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type=Heuristic.manhattan,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1],
                                time_out=1,
                                )
        config = solver.get_config()
        logger.log_info(config)
        self.assertEqual(dimension, (config['n'], config['m']))
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(inf)
        self.assertEqual(dimension, puzzle.dimension())
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        self.assertTrue(solution.failed())
        self.assertTrue(str(solution).find('Exceeded timeout') >= 0)
        logger.log_info(solution)
        solution = solver.solve(puzzle=puzzle, time_out=300)
        self.assertFalse(solution.failed())
        logger.log_info(solution)

    def test_deep_reinforcement_learning_solver(self):
        logger = Loggable(name='test_deep_reinforcement_learning_solver')
        puzzle_type = Puzzle.sliding_puzzle
        dimension = (2, 2)
        # we learn first
        model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='test_deep_reinforcement_learning_solver')
        remove_file(model_file_name)
        learner = DeepReinforcementLearner(puzzle_type=Puzzle.sliding_puzzle,
                                           learning_file_name=model_file_name,
                                           solver_type=Solver.astar,
                                           n=dimension[0],
                                           m=dimension[1],
                                           nb_cpus=1,
                                           network_type=DeepLearning.fully_connected_net,
                                           layers_description=(16, 8),
                                           nb_epochs=1000,
                                           one_hot_encoding=True,
                                           nb_shuffles=12,
                                           max_target_not_increasing_epochs_pct=0.5,
                                           learning_rate=1e-2,
                                           nb_sequences=1)
        logger.log_info(learner.get_config())
        learner.learn()
        # Then we use this learning to solve
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type=Heuristic.deep_learning,
                                model_file_name=model_file_name,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1])
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(inf)
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        logger.log_info(solution)
        remove_file(model_file_name)

########################################################################################################################
