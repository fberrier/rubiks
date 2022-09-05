# ###################################################################
from math import inf
# ###################################################################
from rubiks . deeplearning . deeplearning import DeepLearning
from rubiks . learners . learner import Learner
from rubiks . learners . deeplearner import DeepLearner
from rubiks . puzzle . puzzle import Puzzle
from rubiks . puzzle . trainingdata import TrainingData
# ###################################################################
if '__main__' == __name__:
    puzzle_type = Puzzle . sliding_puzzle
    n=5
    m=2
    """ Generate training data - 100 sequences of fully
    solved perfectly shuffled puzzles .
    """
    nb_cpus = 1
    time_out = 600
    nb_shuffles = inf
    nb_sequences = 5 # 100
    TrainingData ( ** globals () ) . generate ( ** globals () )
    exit()
    """ DL learner """
    action_type = Learner . do_learn
    learner_type = Learner . deep_learner
    nb_epochs = 999
    learning_rate = 1e-3
    optimiser = DeepLearner . rms_prop
    scheduler = DeepLearner . exponential_scheduler
    gamma_scheduler = 0.9999
    save_at_each_epoch = False
    threshold = 0.01
    training_data_freq = 100
    high_target = nb_shuffles + 1
    training_data_from_data_base = True
    nb_shuffles_min = 20
    nb_shuffles_max = 50
    nb_sequences = 50
    """ ... and its network config """
    network_type = DeepLearning . fully_connected_net
    layers_description = ( 100 , 50 , 10 )
    one_hot_encoding = True
    """ Kick - off the Deep Learner """
    learning_file_name = Learner . factory ( ** globals () ) . get_model_name ()
    Learner . factory ( ** globals () ) . action ()
    """ Plot its learning """
    action_type = Learner . do_plot
    Learner . factory ( ** globals () ) . action ()
# ##################################################################