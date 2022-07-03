########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic
from rubiks.heuristics.heuristic import Heuristic
from rubiks.heuristics.perfectheuristic import PerfectHeuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import get_model_file_name, get_shuffles_file_name, get_performance_file_name
########################################################################################################################


if '__main__' == __name__:
    """ Just create a logger to print some stuff in this script """
    logger = Loggable(name=__file__)
    """ What we want to do {Solver.do_solve,
                            Solver.do_performance_test,
                            Solver.do_plot,
                            Solver.do_cleanup_performance_file,
                            Solver.do_cleanup_shuffles_file,
                            } """
    action_type = Solver.do_solve
    """ What puzzle """
    puzzle_type = Puzzle.sliding_puzzle
    tiles = [[8, 6, 7], [2, 5, 4], [3, 0, 1]]
    n = 3
    m = 3
    dimension = Puzzle.factory(**globals()).dimension()
    """ How much to shuffle """
    nb_shuffles = 0
    """ For performance test """
    nb_samples = 1000
    min_nb_shuffles = 0
    max_nb_shuffles = 50
    step_nb_shuffles = 2
    add_perfect_shuffle = True
    nb_cpus = 5
    performance_file_name = get_performance_file_name(puzzle_type, dimension)
    shuffles_file_name = get_shuffles_file_name(puzzle_type, dimension)
    append = True
    """ For plot """
    performance_metrics = [Solver.pct_solved,
                           Solver.pct_optimal,
                           Solver.avg_run_time,
                           Solver.avg_cost,
                           Solver.max_cost,
                           Solver.avg_expanded_nodes,
                           ]
    fig_size = [20, 12]
    """ Which solver type {Solver.dfs,
                           Solver.bfs,
                           Solver.astar,
                           Solver.naive,
                           } """
    solver_type = Solver.astar
    limit = 10
    time_out = 360
    log_solution = True
    check_optimal = True
    max_consecutive_timeout = 25
    """ Heuristic if a* {Heuristic.manhattan,
                         Heuristic.perfect,
                         Heuristic.deep_learning,
                         } """
    heuristic_type = Heuristic.deep_learning
    """ If deep_learning, what network_type {DeepLearning.fully_connected_net} """
    network_type = DeepLearning.fully_connected_net
    layers_description = (599, 300, 100)
    one_hot_encoding = True
    try:
        if heuristic_type == Heuristic.deep_learning:
            model_file_name = DeepLearning.factory(**globals()).get_model_name()
            logger.log_debug({DeepLearningHeuristic.model_file_name: model_file_name})
        elif heuristic_type == Heuristic.perfect:
            model_file_name = get_model_file_name(puzzle_type,
                                                  dimension,
                                                  Heuristic.perfect)
            logger.log_debug({PerfectHeuristic.model_file_name: model_file_name})
    except ValueError:
        model_file_name = None
    Solver.factory(**globals()).action()

########################################################################################################################

