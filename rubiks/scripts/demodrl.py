########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from sys import argv
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.utils.utils import is_windows, g_not_a_pkl_file
########################################################################################################################


def main():
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-e', type=int, default=None)
    parser.add_argument('-u', type=int, default=None)
    parser.add_argument('-nb_shuffles', type=int, default=100)
    parser.add_argument('-nb_sequences', type=int, default=1)
    parser.add_argument('-verbose', type=bool, default=False)
    parser.add_argument('-use_cuda', type=bool, default=True)
    parser.add_argument('-action', type=str, default='learn', choices=['learn', 'plot'])
    parser.add_argument('-model_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-learning_file_name', type=str, default=g_not_a_pkl_file)
    parser = parser.parse_args()
    network_type = DeepLearning.fully_connected_net
    nb_epochs = parser.e
    update_target_network_frequency = parser.u
    n = parser.n
    m = parser.m if parser.m is not None else n
    learner = DeepReinforcementLearner(SlidingPuzzle,
                                       nb_epochs=nb_epochs,
                                       nb_shuffles=parser.nb_shuffles,
                                       nb_sequences=parser.nb_sequences,
                                       n=n,
                                       m=m,
                                       network_type=network_type,
                                       learning_rate=1e-3,
                                       update_target_network_frequency=update_target_network_frequency,
                                       verbose=parser.verbose,
                                       use_cuda=parser.use_cuda,
                                       layers=(3**4, 9, 3),
                                       )
    if 'learn' == parser.action:
        learner.learn()
        learner.save(model_file_name=parser.model_file_name,
                     learning_file_name=parser.learning_file_name)
    elif 'plot' == parser.action:
        learner.plot_learning(learning_file_name=parser.learning_file_name)
    else:
        raise ValueError('Unknown action \'%s\'' % parser.action)
    
########################################################################################################################


if '__main__' == __name__:
    if is_windows():
        command_line_args = "3 -e=1500 -u=50 -nb_sequences=50 -nb_shuffles=30"
        command_line_args += " -action=learn"
        command_line_args += " -model_file_name=models/demodrl_8_puzzle_fully_connected_net.pkl"
        command_line_args += " -learning_file_name=models/demodrl_8_puzzle_fully_connected_net_convergence_data.pkl"
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
