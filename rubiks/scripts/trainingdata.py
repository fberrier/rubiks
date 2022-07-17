########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.trainingdata import TrainingData
########################################################################################################################


if '__main__' == __name__:
    logger = Loggable(name=__file__)
    puzzle_type = Puzzle.sliding_puzzle
    n = 4
    m = 4
    nb_cpus = 1
    nb_sequences = nb_cpus * 2
    time_out = 3600
    training_data = TrainingData(**globals())
    repeat = 1
    for nb_shuffles in [1]:
        training_data.generate(nb_shuffles=nb_shuffles,
                               nb_sequences=nb_sequences,
                               repeat=repeat)

########################################################################################################################

