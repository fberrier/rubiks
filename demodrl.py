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
    solver = solver(SlidingPuzzle,
                    nb_epochs=1,
                    nb_shuffles=2,
                    nb_sequences=2,
                    n=2,
                    m=2,
                    **kw_args)
    solver.learn()
########################################################################################################################


if '__main__' == __name__:
    main()

########################################################################################################################
