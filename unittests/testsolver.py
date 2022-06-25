########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
from os import remove
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.heuristics.heuristic import Heuristic
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
from rubiks.learners.perfectlearner import PerfectLearner
from rubiks.utils.utils import get_model_file_name
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
        try:
            remove(model_file_name)
        except FileNotFoundError:
            pass
        learner = PerfectLearner(puzzle_type=Puzzle.sliding_puzzle,
                                 data_base_file_name=model_file_name,
                                 solver_type=Solver.astar,
                                 heuristic_type=Heuristic.manhattan,
                                 time_out=2,
                                 n=dimension[0],
                                 m=dimension[1])
        logger.log_info(learner.get_config())
        learner.learn()
        # Then we use this learning to solve
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type='perfect',
                                model_file_name=model_file_name,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1])
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(10)
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        logger.log_info(solution)

########################################################################################################################
