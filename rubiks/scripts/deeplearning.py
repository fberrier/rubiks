########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from multiprocessing import cpu_count
from sys import argv
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.utils.utils import is_windows, g_not_a_pkl_file, training_file_name, model_file_name
########################################################################################################################


def main():
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-nb_epochs', type=int, default=None)
    parser.add_argument('-update_target_network_frequency', type=int, default=None)
    parser.add_argument('-update_target_network_threshold', type=float, default=None)
    parser.add_argument('-max_nb_target_network_update', type=int, default=None)
    parser.add_argument('-max_target_not_increasing_epochs_pct', type=float, default=None)
    parser.add_argument('-max_target_uptick', type=float, default=None)
    parser.add_argument('-nb_shuffles', type=int, default=100)
    parser.add_argument('-nb_sequences', type=int, default=1)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--use_cuda', default=False, action='store_true')
    parser.add_argument('-action', type=str, default='learn', choices=['learn', 'train', 'plot'])
    parser.add_argument('-model_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-learning_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--one_hot_encoding', default=False, action='store_true')
    parser.add_argument('-nb_cpus', type=int, default=int(cpu_count() / 2))
    parser = parser.parse_args()
    network_type = DeepLearning.fully_connected_net
    n = parser.n
    m = parser.m if parser.m is not None else n
    learner = DeepReinforcementLearner(SlidingPuzzle,
                                       nb_epochs=parser.nb_epochs,
                                       nb_shuffles=parser.nb_shuffles,
                                       nb_sequences=parser.nb_sequences,
                                       n=n,
                                       m=m,
                                       network_type=network_type,
                                       learning_rate=1e-4,
                                       update_target_network_frequency=parser.update_target_network_frequency,
                                       update_target_network_threshold=parser.update_target_network_threshold,
                                       max_nb_target_network_update=parser.max_nb_target_network_update,
                                       max_target_not_increasing_epochs_pct=parser.max_target_not_increasing_epochs_pct,
                                       max_target_uptick=parser.max_target_uptick,
                                       verbose=parser.verbose,
                                       use_cuda=parser.use_cuda,
                                       layers=tuple(parser.layers),
                                       one_hot_encoding=parser.one_hot_encoding,
                                       nb_cpus=parser.nb_cpus,
                                       )
    if parser.action in ['learn', 'train']:
        learner.learn()
        learner.save(model_file_name=parser.model_file_name,
                     learning_file_name=parser.learning_file_name)
    elif 'plot' == parser.action:
        learner.plot_learning(learning_file_name=parser.learning_file_name)
    else:
        raise ValueError('Unknown action \'%s\'' % parser.action)
    
########################################################################################################################


if '__main__' == __name__:
    PuzzleType = SlidingPuzzle
    dimension = (3, 3)
    action = 'plot'
    nb_cpus = 12
    layers = ('600', '300', '100')
    one_hot_encoding = True
    if is_windows():
        command_line_args = "%d -m=%d" % dimension
        command_line_args += " -nb_epochs=3000"
        command_line_args += " -nb_sequences=100 -nb_shuffles=100"
        command_line_args += " -update_target_network_frequency=100"
        command_line_args += " -update_target_network_threshold=0.002"
        command_line_args += " -max_nb_target_network_update=50"
        command_line_args += " -max_target_not_increasing_epochs_pct=0.25"
        command_line_args += " -max_target_uptick=0.01"
        command_line_args += " -action=%s" % action
        command_line_args += " -nb_cpus=%d" % nb_cpus
        model_name = 'fully_connected_net_' + '_'.join(layers)
        if one_hot_encoding:
            model_name += '_one_hot_encoding'
            command_line_args += " --one_hot_encoding"
        model_file_name = model_file_name(puzzle_type=PuzzleType,
                                          dimension=dimension,
                                          model_name=model_name)
        command_line_args += " -model_file_name=%s" % model_file_name
        learning_file_name = training_file_name(puzzle_type=PuzzleType,
                                                dimension=dimension,
                                                model_name=model_name)
        command_line_args += " -learning_file_name=%s" % learning_file_name
        command_line_args += " --layers %s" % ' '.join(layers)
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
