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


def performance(**kw_args):
    action_type = Solver.do_performance_test
    """ What puzzle """
    puzzle_type = Puzzle.rubiks_cube
    n = 3
    dimension = Puzzle.factory(**locals()).dimension()
    """ How much to shuffle """
    nb_shuffles = 0
    """ For performance test """
    solver_type = Solver.bfs
    nb_samples = 50
    min_nb_shuffles = 0
    max_nb_shuffles = 40
    step_nb_shuffles = 2
    add_perfect_shuffle = True
    nb_cpus = 5
    chunk_size = 0
    performance_file_name = get_performance_file_name(puzzle_type, dimension)
    shuffles_file_name = get_shuffles_file_name(puzzle_type, dimension)
    append = True
    verbose = True
    do_not_reattempt_failed = False
    time_out = 7200
    log_solution = False
    check_optimal = False
    max_consecutive_timeout = 1000
    if kw_args.get('heuristic_type', None) in [Heuristic.deep_learning,
                                               Heuristic.deep_q_learning]:
        model_file_name = Learner.factory(**{**locals(), **kw_args}).get_model_name()
    loc = 'upper center'   # 'upper center'
    performance_metrics = [Solver.pct_solved,
                           Solver.optimality_score,
                           Solver.median_cost,
                           Solver.max_cost,
                           Solver.median_run_time,
                           Solver.median_expanded_nodes,
                           ]
    show_title = True
    lite_title = True
    plot_abbreviated_names = True
    kw_args = {**locals(), **kw_args}
    Solver.factory(**kw_args).action()

########################################################################################################################


if '__main__' == __name__:
    action_type = Solver.do_performance_test
    fig_size = (12, 12)
    if action_type == Solver.do_plot:
        exclude_solver_names = ['122seq_40shf', '500seq_12shf']
        performance(**locals())
    elif action_type == Solver.do_performance_test:
        nb_samples = 20
        nb_cpus = 4
        performance(solver_type=Solver.astar,
                    heuristic_type=Heuristic.deep_learning,
                    learner_type=Learner.deep_reinforcement_learner,
                    time_out=1200,
                    add_perfect_shuffle=False,
                    max_nb_shuffles=4,
                    step_nb_shuffles=2,
                    nb_sequences=10000, #500,
                    nb_shuffles=12,
                    nb_epochs=10000,
                    layers_description=(600, 300, 100),
                    training_data_every_epoch=False, #True,
                    cap_target_at_network_count=False,
                    one_hot_encoding=True,
                    drop_out=0.,
                    network_type=DeepLearning.fully_connected_net,
                    learning_rate=1e-3,
                    scheduler=DeepReinforcementLearner.exponential_scheduler,
                    gamma_scheduler=0.9999,
                    **locals())
        exit()
        performance(solver_type=Solver.astar,
                    heuristic_type=Heuristic.deep_learning,
                    learner_type=Learner.deep_learner,
                    time_out=3600 + 7200,
                    add_perfect_shuffle=False,
                    min_nb_shuffles=10,
                    max_nb_shuffles=12,             #  ->  one more to re-try for nb_shuffle=10 #  -> try up to 12 to start with
                    step_nb_shuffles=2,
                    nb_sequences=100,
                    nb_shuffles=32,
                    nb_epochs=100000,
                    layers_description=(600, 300, 100),
                    training_data_every_epoch=True,
                    cap_target_at_network_count=True,
                    one_hot_encoding=True,
                    drop_out=0.,
                    network_type=DeepLearning.fully_connected_net,
                    learning_rate=1e-3,
                    scheduler=DeepReinforcementLearner.exponential_scheduler,
                    gamma_scheduler=0.9999,
                    nb_shuffles_min=1,
                    nb_shuffles_max=32,
                    **locals())
        exit()
        performance(solver_type=Solver.bfs,
                    add_perfect_shuffle=False,
                    max_nb_shuffles=5,
                    step_nb_shuffles=1,
                    **locals())
        exit()
        performance(solver_type=Solver.astar,
                    heuristic_type=Heuristic.deep_learning,
                    learner_type=Learner.deep_learner,
                    add_perfect_shuffle=False,
                    max_nb_shuffles=4,
                    step_nb_shuffles=4,
                    nb_sequences=1000,
                    nb_shuffles=50,
                    nb_epochs=100000,
                    layers_description=(4096, 2048, 512),
                    training_data_every_epoch=False,
                    cap_target_at_network_count=True,
                    one_hot_encoding=True,
                    drop_out=0.,
                    network_type=DeepLearning.fully_connected_net,
                    learning_rate=1e-2,
                    scheduler=DeepReinforcementLearner.exponential_scheduler,
                    gamma_scheduler=0.9999,
                    nb_shuffles_min=1,
                    nb_shuffles_max=32,
                    **locals())
        performance(solver_type=Solver.astar,
                    heuristic_type=Heuristic.kociemba,
                    **locals())
        performance(solver_type=Solver.kociemba,
                    **locals())

########################################################################################################################

