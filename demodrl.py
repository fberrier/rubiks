########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
########################################################################################################################
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.learners.drl import DRL
from rubiks.utils.utils import pprint
########################################################################################################################


def main():
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-e', type=int, default=None)
    parser.add_argument('-u', type=int, default=None)
    parser.add_argument('-nb_shuffles', type=int, default=100)
    parser.add_argument('-nb_sequences', type=int, default=1)
    parser.add_argument('-verbose', type=bool, default=True)
    parser = parser.parse_args()
    network_type = DeepLearning.fully_connected_net
    nb_epochs = parser.e
    update_target_network_frequency = parser.u
    n = parser.n
    m = parser.m
    learner = DRL(SlidingPuzzle,
                  nb_epochs=nb_epochs,
                  nb_shuffles=parser.nb_shuffles,
                  nb_sequences=parser.nb_sequences,
                  n=n,
                  m=m,
                  network_type=network_type,
                  learning_rate=1e-3,
                  update_target_network_frequency=update_target_network_frequency,
                  verbose=parser.verbose)
    learner.learn()
    learner.save(data_base='demodrl_%d_%d_%s.pkl' % (n, m, network_type))
    
########################################################################################################################


if '__main__' == __name__:
    main()

########################################################################################################################
