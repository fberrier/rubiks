########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.solvers.drl import DRLSolver
from rubiks.utils.utils import pprint
########################################################################################################################


def main():
    solver = DRLSolver
    kw_args = {'network_config': {'network_type': 'fully_connected_net'}}
    nb_epochs = 1000
    update_target_network_frequency = 100
    solver = solver(SlidingPuzzle,
                    nb_epochs=nb_epochs,
                    nb_shuffles=25,
                    nb_sequences=1,
                    n=2,
                    m=2,
                    learning_rate=1e-3,
                    update_target_network_frequency=update_target_network_frequency,
                    **kw_args)
    solver.learn()
########################################################################################################################


if '__main__' == __name__:
    main()

########################################################################################################################
