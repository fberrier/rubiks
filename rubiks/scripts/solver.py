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


if '__main__' == __name__:
    """ Just create a logger to print some stuff in this script """
    logger = Loggable(name=__file__)
    """ What we want to do {Solver.do_solve,
                            Solver.do_performance_test,
                            Solver.do_plot,
                            Solver.do_cleanup_performance_file,
                            Solver.do_cleanup_shuffles_file,
                            } """
    action_type = Solver.do_performance_test
    """ What puzzle """
    puzzle_type = Puzzle.sliding_puzzle
    n = 4
    m = 4
    dimension = Puzzle.factory(**globals()).dimension()
    """ How much to shuffle """
    nb_shuffles = 0
    """ For performance test """
    nb_samples = 1000
    min_nb_shuffles = 31
    max_nb_shuffles = 40
    step_nb_shuffles = 1
    add_perfect_shuffle = False
    nb_cpus = 16
    performance_file_name = get_performance_file_name(puzzle_type, dimension)
    shuffles_file_name = get_shuffles_file_name(puzzle_type, dimension)
    append = True
    """ For plot """
    performance_metrics = [Solver.pct_solved,
                           Solver.pct_optimal,
                           Solver.median_run_time,
                           Solver.median_cost,
                           Solver.max_cost,
                           Solver.median_expanded_nodes,
                           ]
    fig_size = [20, 12]
    """ Which solver type {Solver.dfs,
                           Solver.bfs,
                           Solver.astar,
                           Solver.naive,
                           } """
    solver_type = Solver.astar
    limit = 12
    time_out = 1200
    log_solution = True
    check_optimal = True
    max_consecutive_timeout = 25
    """ Heuristic if a* {Heuristic.manhattan,
                         Heuristic.perfect,
                         Heuristic.deep_learning,
                         } """
    heuristic_type = Heuristic.manhattan
    """ If manhattan """
    plus = True
    """ If deep_learning, what network_type {DeepLearning.fully_connected_net} """
    network_type = DeepLearning.fully_connected_net
    layers_description = (600, 300, 100)
    nb_epochs = 10000
    nb_sequences = 100
    nb_shuffles = 100
    scheduler = DeepReinforcementLearner.gamma_scheduler
    training_data_every_epoch = False
    cap_target_at_network_count = True
    one_hot_encoding = True
    drop_out = 0.
    try:
        if heuristic_type == Heuristic.deep_learning:
            learner_type = Learner.deep_reinforcement_learner
            model_file_name = Learner.factory(**globals()).get_model_name()
            logger.log_info({DeepLearningHeuristic.model_file_name: model_file_name})
        elif heuristic_type == Heuristic.perfect:
            model_file_name = get_model_file_name(puzzle_type,
                                                  dimension,
                                                  Heuristic.perfect)
            logger.log_info({PerfectHeuristic.model_file_name: model_file_name})
    except ValueError as error:
        logger.log_error(error)
        model_file_name = None
    Solver.factory(**globals()).action()

########################################################################################################################

