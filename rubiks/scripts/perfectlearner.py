########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from math import inf
from sys import argv
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.learners.perfectlearner import PerfectLearner
from rubiks.utils.utils import is_windows, g_not_a_pkl_file, model_file_name
########################################################################################################################


def main():
    """ This script generates a random Sliding Puzzle and
    attempts to solve it using a solver of your choice """
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-time_out', default=inf)
    parser.add_argument('-nb_cpus', type=int, default=12)
    parser.add_argument('-max_puzzles', default=inf)
    parser.add_argument('-regular_save', default=1000)
    parser.add_argument('-cpu_multiplier', default=10)
    parser.add_argument('-puzzle_type', type=str, default=None,
                        choices=[Puzzle.sliding_puzzle_tag, Puzzle.rubiks_cube_tag])
    parser.add_argument('-data_base_file_name', type=str, default=g_not_a_pkl_file)
    parser = parser.parse_args()
    kw_args = {'n': parser.n,
               'm': parser.m,
               'puzzle_type': parser.puzzle_type,
               'data_base_file_name': parser.data_base_file_name,
               'nb_cpus': parser.nb_cpus,
               'max_puzzles': parser.max_puzzles,
               'time_out': parser.time_out,
               'cpu_multiplier': parser.cpu_multiplier}
    learner = PerfectLearner(**kw_args)
    learner.learn()

########################################################################################################################


if '__main__' == __name__:
    puzzle_type = 'sliding_puzzle'
    dimension = (2, 2)
    time_out = 300
    nb_cpus = 20
    cpu_multiplier = 5
    max_puzzles = 100000
    regular_save = 10000
    if is_windows():
        command_line_args = "%d -m=%d" % dimension
        command_line_args += " -puzzle_type=%s" % puzzle_type
        command_line_args += " -time_out=%d" % time_out
        command_line_args += " -nb_cpus=%d" % nb_cpus
        command_line_args += " -cpu_multiplier=%d" % cpu_multiplier
        command_line_args += " -max_puzzles=%d" % max_puzzles
        command_line_args += " -regular_save=%d" % regular_save
        command_line_args += " -data_base_file_name=%s" % (model_file_name(puzzle_type=puzzle_type,
                                                                           dimension=dimension,
                                                                           model_name='perfect'))
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
