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
    n = 2
    dimension = Puzzle.factory(**locals()).dimension()
    """ How much to shuffle """
    nb_shuffles = 0
    """ For performance test """
    solver_type = Solver.bfs
    nb_samples = 5
    min_nb_shuffles = 0
    max_nb_shuffles = 20
    step_nb_shuffles = 2
    add_perfect_shuffle = True
    nb_cpus = 1
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
    plot_abbreviated_names = True
    labels_at_top = True
    show_title = True
    lite_title = True
    plot_abbreviated_names = True
    kw_args = {**locals(), **kw_args}
    Solver.factory(**kw_args).action()

########################################################################################################################


if '__main__' == __name__:
    logger = Loggable(name='Perf22RC')
    action_type = Solver.do_performance_test
    fig_size = (12, 11)
    if action_type == Solver.do_plot:
        exclude_solver_names = ['1000seq_20shf']
        performance(**locals())
    elif action_type == Solver.do_performance_test:
        nb_samples = 100
        nb_cpus = 8
        for counter in [20]:
            logger.log_info('counter: ', counter)
            performance(solver_type=Solver.astar,
                        heuristic_type=Heuristic.deep_q_learning,
                        learner_type=Learner.deep_q_learner,
                        time_out=3600,
                        add_perfect_shuffle=counter == 22,
                        min_nb_shuffles=0 if counter > 20 else counter,
                        max_nb_shuffles=0 if counter > 20 else counter,
                        nb_sequences=10000,
                        nb_shuffles=15,
                        chunk_size=1,
                        nb_epochs=10000,
                        layers_description=(600, 300, 100),
                        training_data_every_epoch=False,
                        cap_target_at_network_count=False,
                        one_hot_encoding=True,
                        drop_out=0.,
                        network_type=DeepLearning.fully_connected_net,
                        learning_rate=1e-2,
                        scheduler=DeepReinforcementLearner.exponential_scheduler,
                        gamma_scheduler=0.9999,
                        **locals())
        exit()








        performance(solver_type=Solver.astar,
                    heuristic_type=Heuristic.deep_learning,
                    learner_type=Learner.deep_reinforcement_learner,
                    time_out=600,
                    add_perfect_shuffle=True,
                    max_nb_shuffles=20,
                    nb_sequences=1000,
                    nb_shuffles=20,
                    nb_epochs=100000,
                    layers_description=(600, 300, 100),
                    training_data_every_epoch=False,
                    cap_target_at_network_count=True,
                    one_hot_encoding=True,
                    drop_out=0.,
                    network_type=DeepLearning.fully_connected_net,
                    learning_rate=1e-2,
                    scheduler=DeepReinforcementLearner.exponential_scheduler,
                    gamma_scheduler=0.9999,
                    **locals())
        exit()
        performance(solver_type=Solver.mcts,
                    c=10,
                    trim_tree=True,
                    add_perfect_shuffle=False,
                    max_nb_shuffles=6,
                    heuristic_type=Heuristic.deep_q_learning,
                    learner_type=Learner.deep_q_learner,
                    nb_sequences=1000,
                    nb_shuffles=20,
                    nb_epochs=100000,
                    layers_description=(600, 300, 100),
                    training_data_every_epoch=False,
                    cap_target_at_network_count=True,
                    one_hot_encoding=True,
                    drop_out=0.,
                    network_type=DeepLearning.fully_connected_net,
                    learning_rate=1e-2,
                    scheduler=DeepReinforcementLearner.exponential_scheduler,
                    gamma_scheduler=0.9999,
                    **locals())
        exit()
        performance(solver_type=Solver.bfs,
                    add_perfect_shuffle=False,
                    max_nb_shuffles=6,
                    step_nb_shuffles=1,
                    **locals())
        performance(solver_type=Solver.astar,
                    heuristic_type=Heuristic.deep_learning,
                    learner_type=Learner.deep_learner,
                    nb_sequences=1000,
                    nb_shuffles=30,
                    nb_epochs=100000,
                    layers_description=(600, 300, 100),
                    training_data_every_epoch=False,
                    cap_target_at_network_count=True,
                    one_hot_encoding=True,
                    drop_out=0.,
                    network_type=DeepLearning.fully_connected_net,
                    learning_rate=1e-2,
                    scheduler=DeepReinforcementLearner.exponential_scheduler,
                    gamma_scheduler=0.9999,
                    nb_shuffles_min=1,
                    nb_shuffles_max=14,
                    **locals())
        performance(solver_type=Solver.kociemba,
                    **locals())
        performance(solver_type=Solver.astar,
                    heuristic_type=Heuristic.kociemba,
                    **locals())
        performance(solver_type=Solver.astar,
                    heuristic_type=Heuristic.deep_q_learning,
                    learner_type=Learner.deep_q_learner,
                    nb_sequences=1000,
                    nb_shuffles=20,
                    nb_epochs=100000,
                    layers_description=(600, 300, 100),
                    training_data_every_epoch=False,
                    cap_target_at_network_count=True,
                    one_hot_encoding=True,
                    drop_out=0.,
                    network_type=DeepLearning.fully_connected_net,
                    learning_rate=1e-2,
                    scheduler=DeepReinforcementLearner.exponential_scheduler,
                    gamma_scheduler=0.9999,
                    **locals())

########################################################################################################################

