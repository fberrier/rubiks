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
    puzzle_type = Puzzle.watkins_cube
    n = 2
    m = 2
    nb_cpus = 10
    nb_sequences = nb_cpus * 25
    time_out = 3600
    training_data = TrainingData(**globals())
    repeat = 10
    for nb_shuffles in [10, 20, 30, 40, 50, inf]:
        training_data.generate(nb_shuffles=nb_shuffles,
                               nb_sequences=nb_sequences,
                               repeat=repeat)

########################################################################################################################

