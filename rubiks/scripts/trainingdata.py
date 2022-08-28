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
    n = 3
    nb_cpus = 1
    chunk_size=0
    verbose=False
    nb_sequences = nb_cpus * 1000
    training_data = TrainingData(**globals())
    repeat = 10
    for nb_shuffles in [inf]:
        training_data.generate(nb_shuffles=nb_shuffles,
                               nb_sequences=nb_sequences,
                               repeat=repeat)

########################################################################################################################

