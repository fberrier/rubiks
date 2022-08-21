########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic
from rubiks.heuristics.heuristic import Heuristic
from rubiks.heuristics.perfectheuristic import PerfectHeuristic
from rubiks.learners.learner import Learner
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import get_model_file_name, get_shuffles_file_name, get_performance_file_name
########################################################################################################################
from math import inf

if '__main__' == __name__:
    """ Just create a logger to print some stuff in this script """
    logger = Loggable(name=__file__)
    """ What puzzle """
    puzzle_type = Puzzle.sliding_puzzle
    tiles = [[4, 1, 2], [0, 8, 3], [5, 7, 6]]
    puzzle = Puzzle.factory(**globals())
    logger.log_info(puzzle)
    solver_type = Solver.mcts
    time_out = inf
    log_solution = False
    check_optimal = False
    max_consecutive_timeout = 100
    """ Heuristic if a* {Heuristic.manhattan,
                         Heuristic.perfect,
                         Heuristic.deep_learning,
                         } """
    heuristic_type = Heuristic.deep_q_learning
    """ If MCTS """
    c = 1
    nu = 1
    learner_type = Learner.deep_q_learner
    model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                          dimension=(3, 3),
                                          model_name='test_deep_q_learning_solver')
    solution = Solver.factory(**globals()).solve(puzzle)
    logger.log_info(solution)

########################################################################################################################

