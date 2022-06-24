########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from sys import argv
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.learners.perfectlearner import PerfectLearner, Learner
from rubiks.utils.utils import is_windows, get_model_file_name
########################################################################################################################


def main():
    """ This script generates a random Sliding Puzzle and
    attempts to solve it using a solver of your choice """
    parser = ArgumentParser()
    SlidingPuzzle.populate_parser(parser)
    PerfectLearner.populate_parser(parser)
    parser = parser.parse_args()
    learner = Learner.factory(**parser.__dict__)
    learner.learn()

########################################################################################################################


if '__main__' == __name__:
    puzzle_type = Puzzle.sliding_puzzle
    dimension = (3, 3)
    time_out = 300
    nb_cpus = 20
    cpu_multiplier = 5
    max_puzzles = 100000
    regular_save = 10000
    if is_windows():
        command_line_args = " -n=%d -m=%d" % dimension
        command_line_args += " -puzzle_type=%s" % puzzle_type
        command_line_args += " -time_out=%d" % time_out
        command_line_args += " -nb_cpus=%d" % nb_cpus
        command_line_args += " -cpu_multiplier=%d" % cpu_multiplier
        command_line_args += " -max_puzzles=%d" % max_puzzles
        command_line_args += " -regular_save=%d" % regular_save
        command_line_args += " -data_base_file_name=%s" % (get_model_file_name(puzzle_type=puzzle_type,
                                                                               dimension=dimension,
                                                                               model_name='perfect'))
        argv.extend(command_line_args.strip().split(' '))
    main()

########################################################################################################################
