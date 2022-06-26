########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from sys import argv
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.learners.perfectlearner import PerfectLearner
from rubiks.heuristics.heuristic import Heuristic
from rubiks.utils.utils import is_windows, get_model_file_name
########################################################################################################################


def main(line):
    PerfectLearner.from_command_line(line).action()

########################################################################################################################


if "__main__" == __name__:
    puzzle_type = Puzzle.sliding_puzzle
    dimension = (2, 5)
    time_out = 300
    nb_cpus = 20
    cpu_multiplier = 50
    max_puzzles = 150000
    regular_save = 15000
    after_round_save = True
    flush_timed_out_puzzles = True
    # {random_puzzle_generation, permutation_puzzle_generation}
    puzzle_generation = PerfectLearner.permutation_puzzle_generation
    heuristic_type = Heuristic.manhattan
    # {do_plot, do_cleanup_learning_file, do_learn}
    action_type = PerfectLearner.do_learn
    learning_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                             dimension=dimension,
                                             model_name=PerfectLearner.perfect)
    if is_windows():
        command_line = " -n=%d -m=%d" % dimension
        command_line += " -puzzle_type=%s" % puzzle_type
        command_line += " -time_out=%d" % time_out
        command_line += " -heuristic_type=%s" % heuristic_type
        command_line += " -action_type=%s" % action_type
        command_line += " -puzzle_generation=%s" % puzzle_generation
        command_line += " -nb_cpus=%d" % nb_cpus
        command_line += " -cpu_multiplier=%d" % cpu_multiplier
        command_line += " -max_puzzles=%d" % max_puzzles
        command_line += " -regular_save=%d" % regular_save
        if after_round_save:
            command_line += " --after_round_save"
        if flush_timed_out_puzzles:
            command_line += " --flush_timed_out_puzzles"
        command_line += " -learning_file_name=%s" % learning_file_name
    else:
        command_line = argv
    main(command_line)

########################################################################################################################
