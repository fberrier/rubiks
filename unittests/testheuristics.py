########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf
from unittest import TestCase
########################################################################################################################
from rubiks.heuristics.manhattan import Manhattan
from rubiks.core.loggable import Loggable
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.slidingpuzzle import SlidingPuzzle
from rubiks.solvers.solver import Solver
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

    def manhattan_plus_check(self, name, n, m=None):
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
            solution = manhattan.solve(sliding_puzzle, time_out=inf)
            solution_plus = manhattan_plus.solve(sliding_puzzle, time_out=inf)
            if solution.cost != solution_plus.cost:
                logger.log_error(solution)
                logger.log_error(solution_plus)
                self.assertLessEqual(solution.cost, solution_plus.cost)
            self.assertEqual(solution.cost,
                             solution_plus.cost,
                             'manhattan = %d, manhattan+ = %d for %s' % (solution.cost,
                                                                         solution_plus.cost,
                                                                         sliding_puzzle))
            if 0 == count % 100:
                logger.log_info('Checked ', count, ' puzzles')

    def test_manhattan_distance_2_2_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_2_2_plus', n=2, m=2)

    def test_manhattan_distance_2_3_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_2_2_plus', n=2, m=3)

    def test_manhattan_distance_2_4_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_2_2_plus', n=2, m=4)

    def test_manhattan_distance_3_3_plus(self):
        self.manhattan_plus_check('test_manhattan_distance_2_2_plus', n=3, m=3)

    def test_something(self):
        logger = Loggable(name='test_something')
        sliding_puzzle = Puzzle.factory(puzzle_type=Puzzle.sliding_puzzle,
                                        tiles=[[5, 3, 4], [1, 0, 2]])
        logger.log_info(sliding_puzzle)
        logger.log_info(Manhattan(n=2, m=3).cost_to_go_from_puzzle(sliding_puzzle))
        logger.log_info(Manhattan(n=2, m=3, plus=True).cost_to_go_from_puzzle(sliding_puzzle))



########################################################################################################################
