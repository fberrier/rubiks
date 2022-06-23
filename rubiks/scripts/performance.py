########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pandas import to_pickle, read_pickle, concat
from sys import argv
########################################################################################################################
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.heuristics.heuristic import Heuristic
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import pprint, is_windows, g_not_a_pkl_file, touch
from rubiks.utils.utils import model_file_name, shuffles_file_name, perf_file_name
########################################################################################################################


def main():
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-min_nb_shuffles', type=int, default=None)
    parser.add_argument('-max_nb_shuffles', type=int, default=None)
    parser.add_argument('-step_nb_shuffles', type=int, default=1)
    parser.add_argument('-nb_samples', type=int, default=1)
    parser.add_argument('-time_out', type=int, default=60)
    parser.add_argument('--append', default=False, action='store_true')
    parser.add_argument('-solver_type', type=str, default='bfs', choices=['bfs', 'dfs', 'a*'])
    parser.add_argument('-heuristic_type', type=str, default=Heuristic.manhattan_tag,
                        choices=Heuristic.known_heuristics)
    parser.add_argument('-model_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-action', type=str, default='solve', choices=['solve', 'plot'])
    parser.add_argument('-perf_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('--add_perfect_shuffle', default=False, action='store_true')
    parser.add_argument('-nb_cpus', type=int, default=int(cpu_count() / 2))
    parser.add_argument('-shuffles_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-max_consecutive_timeout', type=int, default=0)
    parser.add_argument('-limit', type=int, default=31)
    parser = parser.parse_args()
    kw_args = {'n': parser.n,
               'm': parser.m,
               'max_consecutive_timeout': parser.max_consecutive_timeout,
               'heuristic_type': heuristic_type,
               'model_file_name': parser.model_file_name,
               'data_base_file_name': parser.model_file_name,
               'limit': parser.limit}
    solver = Solver.factory(solver_type=parser.solver_type,
                            puzzle_type=SlidingPuzzle,
                            **kw_args)
    if parser.action in ['plot']:
        solver.plot_performance(performance_file_name=parser.perf_file_name,
                                solver_name=None)
    else:
        min_nb_shuffles = parser.min_nb_shuffles
        max_nb_shuffles = parser.max_nb_shuffles
        step_nb_shuffles = parser.step_nb_shuffles
        if max_nb_shuffles is None:
            max_nb_shuffles = min_nb_shuffles
        perf_table = solver.performance(max_nb_shuffles=max_nb_shuffles,
                                        nb_samples=parser.nb_samples,
                                        time_out=parser.time_out,
                                        min_nb_shuffles=min_nb_shuffles,
                                        step_nb_shuffles=step_nb_shuffles,
                                        perfect_shuffle=parser.add_perfect_shuffle,
                                        nb_cpus=parser.nb_cpus,
                                        shuffles_file_name=parser.shuffles_file_name)
        pprint(perf_table)
        if parser.perf_file_name and parser.perf_file_name != g_not_a_pkl_file:
            if parser.append:
                try:
                    perf_table = concat((read_pickle(parser.perf_file_name), perf_table))
                except FileNotFoundError:
                    pass
            subset = [Solver.solver_name_tag,
                      Solver.puzzle_type_tag,
                      Solver.puzzle_dimension_tag,
                      Solver.nb_shuffles_tag]
            perf_table = perf_table.drop_duplicates(subset=subset).sort_values(subset)
            touch(parser.perf_file_name)
            to_pickle(perf_table, parser.perf_file_name)
            pprint('Saved ', len(perf_table), ' rows of perf table to ', parser.perf_file_name)
            pprint(perf_table)

########################################################################################################################

    
if '__main__' == __name__:
    if is_windows():
        PuzzleType = SlidingPuzzle
        dimension = (3, 3)
        solver_type = 'a*'
        heuristic_type = 'deep_learning'
        layers = ('600', '300', '100')
        one_hot_encoding = True
        action = 'plot'
        nb_samples = 1000
        min_nb_shuffles = 0
        max_nb_shuffles = 100
        step_nb_shuffles = 10
        add_perfect_shuffle = True
        nb_cpus = 10
        perf_file_name = perf_file_name(puzzle_type=PuzzleType,
                                        dimension=dimension)
        command_line_args = "%d -m=%d" % dimension
        command_line_args += " -min_nb_shuffles=%d -max_nb_shuffles=%d -step_nb_shuffles=%d" % (min_nb_shuffles,
                                                                                                max_nb_shuffles,
                                                                                                step_nb_shuffles)
        command_line_args += " -nb_samples=%d -time_out=120 -nb_cpus=%s " % (nb_samples, nb_cpus)
        command_line_args += "-perf_file_name=%s" % perf_file_name
        if add_perfect_shuffle:
            command_line_args += " --add_perfect_shuffle"
        command_line_args += " --append"
        command_line_args += " -action=%s" % action
        command_line_args += " -solver_type=%s" % solver_type
        command_line_args += " -heuristic_type=%s" % heuristic_type
        command_line_args += " -max_consecutive_timeout=50"
        model = None
        if heuristic_type == 'deep_learning':
            model_name = 'fully_connected_net_' + '_'.join(layers)
            if one_hot_encoding:
                model_name += '_one_hot_encoding'
            model = model_file_name(puzzle_type=PuzzleType,
                                    dimension=dimension,
                                    model_name=model_name)
        elif heuristic_type == 'perfect':
            model = model_file_name(puzzle_type=PuzzleType,
                                    dimension=dimension,
                                    model_name='perfect')
        if model is not None:
            command_line_args += " -model_file_name=%s" % model
        command_line_args += " -shuffles_file_name=%s" % shuffles_file_name(puzzle_type=PuzzleType,
                                                                            dimension=dimension)
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
