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
    action_type = Solver.do_plot
    """ What puzzle """
    puzzle_type = Puzzle.rubiks_cube
    n = 2
    m = 4
    dimension = Puzzle.factory(**globals()).dimension()
    """ How much to shuffle """
    nb_shuffles = 0
    """ For performance test """
    nb_samples = 100
    min_nb_shuffles = 0
    max_nb_shuffles = 45
    step_nb_shuffles = 1
    add_perfect_shuffle = True
    nb_cpus = 8
    chunk_size = 0
    performance_file_name = get_performance_file_name(puzzle_type, dimension)
    shuffles_file_name = get_shuffles_file_name(puzzle_type, dimension)
    append = True
    verbose = True
    do_not_reattempt_failed = False
    #skip = (36,39,85,145,146,161,174,190,) <- these 8 took more than 90 mins for 3x3x3 Rubiks'
    """ For plot """
    loc = 'upper center'   # 'upper center'
    performance_metrics = [Solver.pct_solved,
                           Solver.optimality_score,
                           Solver.median_cost,
                           Solver.max_cost,
                           Solver.median_run_time,
                           Solver.median_expanded_nodes,
                           ]
    plot_abbreviated_names = True
    labels_at_top = True
    show_title = True
    lite_title = True
    #marker_size = 120
    #markers = ['4', 'x', '.', 'x', '.', 'x', '.', 'x', '.', ]
    #colors = ['olive', 'royalblue', 'royalblue', 'darkred', 'darkred', 'goldenrod', 'goldenrod', 'darkcyan', 'darkcyan']
    #exclude_solver_names = ['hattan', 'BFS', 'MCTS', 'Perf', 'Reinf', 'QL', 'Naive']
    exclude_solver_names = ['seen=1.4', '1.2e-07', '1e-07', '9.5']   #  <- for SP 4, 4
    fig_size = (12, 8)
    """ Which solver type {Solver.dfs,
                           Solver.bfs,
                           Solver.astar,
                           Solver.naive,
                           Solver.kociemba,
                           } """
    solver_type = Solver.astar
    c = 100
    trim_tree = True
    limit = 12
    time_out = 7200
    log_solution = False
    check_optimal = False
    max_consecutive_timeout = 100
    """ Heuristic if a* {Heuristic.manhattan,
                         Heuristic.perfect,
                         Heuristic.deep_learning,
                         } """
    heuristic_type = Heuristic.deep_q_learning
    """ If manhattan """
    plus = True
    """ If deep_learning, what network_type {DeepLearning.fully_connected_net,
                                             DeepLearning.convolutional_net} """
    learner_type = Learner.deep_q_learner
    network_type = DeepLearning.fully_connected_net
    layers_description = (600, 300, 100)
    nb_epochs = 100000
    nb_sequences = 150
    nb_shuffles = 75
    nb_shuffles_min = 37
    nb_shuffles_max = 47
    learning_rate = 1e-2
    scheduler = DeepReinforcementLearner.exponential_scheduler
    gamma_scheduler = 0.9999
    training_data_every_epoch = False
    cap_target_at_network_count = True
    one_hot_encoding = True
    drop_out = 0.
    """ Or for convo """
    kernel_size = (2, 2)
    convo_layers_description = (81, 300,)
    parallel_fully_connected_layers_description = (300,)
    fully_connected_layers_description = (600, 300, 100,)
    padding = 0
    try:
        if heuristic_type in [Heuristic.deep_learning, Heuristic.deep_q_learning]:
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

