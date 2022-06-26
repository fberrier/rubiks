########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from sys import argv
########################################################################################################################
from rubiks.puzzle.slidingpuzzle import SlidingPuzzle
from rubiks.puzzle.rubikscube import RubiksCube
from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic, Heuristic
from rubiks.heuristics.perfectheuristic import PerfectHeuristic
from rubiks.heuristics.manhattan import Manhattan
from rubiks.solvers.dfssolver import DFSSolver
from rubiks.solvers.astarsolver import Solver, AStarSolver
from rubiks.utils.utils import is_windows
from rubiks.utils.utils import get_model_file_name, get_shuffles_file_name, get_perf_file_name
########################################################################################################################


def main():
    parser = ArgumentParser()
    SlidingPuzzle.populate_parser(parser)
    RubiksCube.populate_parser(parser)
    DFSSolver.populate_parser(parser)
    AStarSolver.populate_parser(parser)
    DeepLearningHeuristic.populate_parser(parser)
    Manhattan.populate_parser(parser)
    PerfectHeuristic.populate_parser(parser)
    parser = parser.parse_args()
    Solver.factory(**parser.__dict__).action(**parser.__dict__)

########################################################################################################################

    
if '__main__' == __name__:
    if is_windows():
        PuzzleType = SlidingPuzzle
        dimension = (3, 3)
        solver_type = Solver.astar
        heuristic_type = Heuristic.manhattan
        layers = ('600', '300', '100')
        one_hot_encoding = True
        action_type = Solver.do_plot
        nb_samples = 10
        min_nb_shuffles = 0
        max_nb_shuffles = 100
        step_nb_shuffles = 10
        add_perfect_shuffle = True
        nb_cpus = 1
        performance_file_name = get_perf_file_name(puzzle_type=PuzzleType,
                                                   dimension=dimension)
        command_line_args = " -n=%d -m=%d" % dimension
        command_line_args += " -min_nb_shuffles=%d -max_nb_shuffles=%d -step_nb_shuffles=%d" % (min_nb_shuffles,
                                                                                                max_nb_shuffles,
                                                                                                step_nb_shuffles)
        command_line_args += " -nb_samples=%d -time_out=120 -nb_cpus=%s " % (nb_samples, nb_cpus)
        command_line_args += "-performance_file_name=%s" % performance_file_name
        if add_perfect_shuffle:
            command_line_args += " --add_perfect_shuffle"
        command_line_args += " --append"
        command_line_args += " -action_type=%s" % action_type
        command_line_args += " -solver_type=%s" % solver_type
        command_line_args += " -heuristic_type=%s" % heuristic_type
        command_line_args += " -max_consecutive_timeout=50"
        model = None
        if heuristic_type == 'deep_learning':
            model_name = 'fully_connected_net_' + '_'.join(layers)
            if one_hot_encoding:
                model_name += '_one_hot_encoding'
            model = get_model_file_name(puzzle_type=PuzzleType,
                                        dimension=dimension,
                                        model_name=model_name)
        elif heuristic_type == 'perfect':
            model = get_model_file_name(puzzle_type=PuzzleType,
                                        dimension=dimension,
                                        model_name='perfect')
        if model is not None:
            command_line_args += " -model_file_name=%s" % model
        command_line_args += " -shuffles_file_name=%s" % get_shuffles_file_name(puzzle_type=PuzzleType,
                                                                                dimension=dimension)
        argv.extend(command_line_args.strip().split(' '))
    main()

########################################################################################################################
