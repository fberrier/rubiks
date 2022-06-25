########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from sys import argv
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.learners.perfectlearner import PerfectLearner
from rubiks.utils.utils import is_windows, get_model_file_name
########################################################################################################################


def main(line):
    PerfectLearner.from_command_line(line).learn()

########################################################################################################################


if '__main__' == __name__:
    puzzle_type = Puzzle.sliding_puzzle
    dimension = (2, 2)
    time_out = 300
    nb_cpus = 20
    cpu_multiplier = 5
    max_puzzles = 100000
    regular_save = 10000
    data_base_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='perfect')
    if is_windows():
        command_line = " -n=%d -m=%d" % dimension
        command_line += " -puzzle_type=%s" % puzzle_type
        command_line += " -time_out=%d" % time_out
        command_line += " -nb_cpus=%d" % nb_cpus
        command_line += " -cpu_multiplier=%d" % cpu_multiplier
        command_line += " -max_puzzles=%d" % max_puzzles
        command_line += " -regular_save=%d" % regular_save
        command_line += " -data_base_file_name=%s" % data_base_file_name
    else:
        command_line = argv
    main(command_line)

########################################################################################################################
