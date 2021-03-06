########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.heuristics.heuristic import Heuristic
from rubiks.learners.learner import Learner
from rubiks.learners.perfectlearner import PerfectLearner
from rubiks.learners.deeplearner import DeepLearner
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.puzzle.puzzle import Puzzle
from rubiks.utils.utils import get_model_file_name
########################################################################################################################


if '__main__' == __name__:
    """ Just create a logger to print some stuff in this script """
    logger = Loggable(name=__file__)
    """ What we want to do {Learner.do_learn,
                            Learner.do_plot,
                            Learner.do_cleanup_learning_file,
                            } """
    action_type = Learner.do_learn
    """ What puzzle """
    puzzle_type = Puzzle.sliding_puzzle
    n = 4
    m = 4
    dimension = Puzzle.factory(**globals()).dimension()
    """ Which learner_type {Learner.perfect,
                            Learner.deep_reinforcement_learner,
                            } 
    """
    learner_type = Learner.deep_learner
    """ for plot """
    plot_metrics = DeepReinforcementLearner.default_plot_metrics \
        if learner_type is Learner.deep_reinforcement_learner \
        else DeepLearner.default_plot_metrics
    """ If it's a perfect learner config here """
    time_out = 3600
    nb_cpus = 1
    cpu_multiplier = 25
    max_puzzles = nb_cpus * cpu_multiplier * 10000
    regular_save = nb_cpus * cpu_multiplier * 1
    save_timed_out_max_puzzles = 100000
    after_round_save = True
    flush_timed_out_puzzles = True
    save_timed_out = True
    rerun_timed_out = False
    rerun_timed_out_only = False
    abort_after_that_many_consecutive_timed_out = 100
    nb_shuffles_from_goal = 10
    """ puzzle generation process {PerfectLearner.perfect_random_puzzle_generation,
                                   PerfectLearner.permutation_puzzle_generation,
                                   PerfectLearner.random_from_goal_puzzle_generation,
                                   } """
    puzzle_generation = PerfectLearner.permutation_puzzle_generation
    heuristic_type = Heuristic.manhattan
    plus = True
    """ If it's a DRL learner config is here ... """
    nb_epochs = 10000
    nb_sequences = 100
    nb_shuffles = 60
    training_data_every_epoch = False
    cap_target_at_network_count = True
    update_target_network_frequency = 1000
    update_target_network_threshold = 1e-3
    max_nb_target_network_update = 100
    max_target_not_increasing_epochs_pct = 0.5
    max_target_uptick = 0.01
    learning_rate = 1e-3
    scheduler = DeepReinforcementLearner.gamma_scheduler
    gamma_scheduler = 0.9999
    """ DL learner """
    save_at_each_epoch = False
    threshold = 0.01
    training_data_freq = 1000
    high_target = nb_shuffles + 1
    training_data_from_data_base = True
    nb_shuffles_min = 40
    nb_shuffles_max = 60
    """ ... and its network config """
    network_type = DeepLearning.fully_connected_net
    layers_description = (600, 300, 100)
    one_hot_encoding = True
    drop_out = 0.
    """ Or for convo """
    kernel_size = (2, 2)
    convo_layers_description = (81, 300)
    parallel_fully_connected_layers_description = (300,)
    fully_connected_layers_description = (600, 300, 100,)
    padding = 0
    """ either way we make the learning_file_name to save learning results """
    if learner_type == Learner.perfect_learner:
        learning_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                                 dimension=dimension,
                                                 model_name=PerfectLearner.perfect)
    elif learner_type in [Learner.deep_reinforcement_learner,
                          Learner.deep_learner]:
        learning_file_name = Learner.factory(**globals()).get_model_name()
    """ And we fire the action """
    Learner.factory(**globals()).action()

########################################################################################################################
