########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf
from unittest import TestCase
########################################################################################################################
from rubiks.heuristics.manhattan import Manhattan
from rubiks.core.loggable import Loggable
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.heuristics.heuristic import Heuristic
from rubiks.heuristics.deepqlearningheuristic import DeepQLearningHeuristic
from rubiks.learners.deepqlearner import DeepQLearner
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.slidingpuzzle import SlidingPuzzle
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import get_model_file_name, remove_file
########################################################################################################################


class TestHeuristics(TestCase):

    def test_heuristic_factory(self):
        logger = Loggable(name='test_heuristic_factory')
        manhattan = Heuristic.factory(heuristic_type=Heuristic.manhattan,
                                      n=3,
                                      m=3)
        goal = manhattan.get_goal()
        logger.log_info(goal)

    def test_manhattan_distance_3_3(self):
        manhattan = Manhattan(n=3, m=3)
        goal = manhattan.get_goal()
        for _ in range(100):
            for nb_moves in range(1, 50):
                puzzle = goal.apply_random_moves(nb_moves, min_no_loop=nb_moves)
                heuristic_cost = manhattan.cost_to_go_from_puzzle(puzzle)
                # given the heuristic is admissible, and optimal cost is smaller than nb_moves
                # we should have heuristic_cost <= optimal_cost <= nb_moves
                self.assertGreaterEqual(nb_moves, heuristic_cost)

    def test_manhattan_distance_10_10(self):
        manhattan = Manhattan(n=10, m=10)
        goal = manhattan.get_goal()
        for _ in range(100):
            for nb_moves in range(1, 100):
                puzzle = goal.apply_random_moves(nb_moves, min_no_loop=nb_moves)
                heuristic_cost = manhattan.cost_to_go_from_puzzle(puzzle)
                # given the heuristic is admissible, and optimal cost is smaller than nb_moves
                # we should have heuristic_cost <= optimal_cost <= nb_moves
                self.assertGreaterEqual(nb_moves, heuristic_cost)

    def test_manhattan_distance_3_3_plus(self):
        logger = Loggable(name='test_manhattan_distance_3_3_plus')
        sliding_puzzle = Puzzle.factory(puzzle_type=Puzzle.sliding_puzzle,
                                        tiles=[[8, 2, 0], [3, 4, 1], [5, 6, 7]])
        manhattan = Manhattan(n=3, m=3)
        manhattan_plus = Manhattan(n=3, m=3, plus=True)
        cost = manhattan.cost_to_go_from_puzzle(sliding_puzzle)
        cost_plus = manhattan_plus.cost_to_go_from_puzzle(sliding_puzzle)
        logger.log_info({'puzzle': str(sliding_puzzle),
                         'cost': cost,
                         'cost++': cost_plus})
        self.assertEqual(16, cost)
        self.assertEqual(16, cost_plus)

    def manhattan_plus_check(self, name, n, m=None, modulo=100, max_puzzle=inf):
        logger = Loggable(name=name)
        manhattan = Solver.factory(solver_type=Solver.astar,
                                   heuristic_type=Heuristic.manhattan,
                                   puzzle_type=Puzzle.sliding_puzzle,
                                   n=n,
                                   m=m,
                                   plus=False)
        manhattan_plus = Solver.factory(solver_type=Solver.astar,
                                        heuristic_type=Heuristic.manhattan,
                                        puzzle_type=Puzzle.sliding_puzzle,
                                        n=n,
                                        m=m,
                                        plus=True)
        count = 0
        for sliding_puzzle in SlidingPuzzle.generate_all_puzzles(n=n, m=m):
            count += 1
            solution_plus = manhattan_plus.solve(sliding_puzzle, time_out=inf)
            solution = manhattan.solve(sliding_puzzle, time_out=inf)
            if solution.cost != solution_plus.cost:
                logger.log_error(solution)
                logger.log_error(solution_plus)
                self.assertLessEqual(solution.cost, solution_plus.cost)
            self.assertEqual(solution.cost,
                             solution_plus.cost,
                             'manhattan = %d, manhattan+ = %d for %s' % (solution.cost,
                                                                         solution_plus.cost,
                                                                         sliding_puzzle))
            if 0 == count % modulo:
                logger.log_info('Checked', count, 'puzzles')
            if count > max_puzzle:
                break
        logger.log_info('Checked', count, 'puzzles')

    def test_manhattan_distance_2_2_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_2_2_plus', n=2, m=2)

    def test_manhattan_distance_2_3_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_2_2_plus', n=2, m=3)

    def test_manhattan_distance_2_4_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_2_4_plus', n=2, m=4)

    def test_manhattan_distance_2_5_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_2_5_plus', n=2, m=5, modulo=1)

    def test_manhattan_distance_3_3_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_3_3_plus', n=3, m=3, max_puzzle=1)

    def test_manhattan_distance_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_plus', n=3, m=3, max_puzzle=1)

    def test_manhattan_distance_5_5_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_5_5_plus', n=5, m=5, max_puzzle=1)

    def test_something_2_3(self):
        logger = Loggable(name='test_something_2_3')
        sliding_puzzle = Puzzle.factory(puzzle_type=Puzzle.sliding_puzzle,
                                        tiles=[[5, 3, 4], [1, 0, 2]])
        logger.log_info(sliding_puzzle)
        logger.log_info(Manhattan(n=2, m=3).cost_to_go_from_puzzle(sliding_puzzle))
        logger.log_info(Manhattan(n=2, m=3, plus=True).cost_to_go_from_puzzle(sliding_puzzle))
        self.assertEqual(Manhattan(n=2, m=3).cost_to_go_from_puzzle(sliding_puzzle),
                         Manhattan(n=2, m=3, plus=True).cost_to_go_from_puzzle(sliding_puzzle))

    def test_penalty_2(self):
        expected = (1, 2, 3, 4, 5)
        actual = (3, 0, 4, 2, 5)
        self.assertEqual(2, Manhattan.penalty(expected=expected,
                                              actual=actual))

    def test_penalty_4(self):
        expected = (1, 2, 3, 4, 5)
        actual = (4, 3, 2)
        self.assertEqual(4, Manhattan.penalty(expected=expected,
                                              actual=actual))

    def test_deep_q_learning_heuristic(self):
        logger = Loggable(name='test_deep_q_learning_heuristic')
        puzzle_type = Puzzle.sliding_puzzle
        dimension = (2, 2)
        # we learn first
        model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='test_deep_q_learning_heuristic')
        remove_file(model_file_name)
        learner = DeepQLearner(puzzle_type=Puzzle.sliding_puzzle,
                               learning_file_name=model_file_name,
                               solver_type=Solver.astar,
                               n=dimension[0],
                               m=dimension[1],
                               nb_cpus=1,
                               network_type=DeepLearning.fully_connected_net,
                               layers_description=(64, 32),
                               nb_epochs=10000,
                               one_hot_encoding=True,
                               nb_shuffles=12,
                               max_target_not_increasing_epochs_pct=0.5,
                               max_target_uptick=0.01,
                               max_nb_target_network_update=10,
                               update_target_network_threshold=1e-5,
                               learning_rate=1e-2,
                               nb_sequences=1)
        logger.log_info(learner.get_config())
        learner.learn()
        # Then we use this learning to solve
        heuristic = DeepQLearningHeuristic(**learner.get_config(), model_file_name=model_file_name)
        puzzle = Puzzle.factory(**learner.get_config()).apply_random_moves(inf)
        logger.log_info(puzzle)
        logger.log_info(heuristic.cost_to_go_from_puzzle_impl(puzzle))
        logger.log_info(heuristic.optimal_actions(puzzle))
        remove_file(model_file_name)


########################################################################################################################
