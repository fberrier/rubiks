########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pandas import to_pickle, read_pickle, concat
from sys import argv
########################################################################################################################
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import pprint, is_windows, g_not_a_pkl_file, file_name
########################################################################################################################


def main():
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-r_min', type=int, default=None)
    parser.add_argument('-r_max', type=int, default=None)
    parser.add_argument('-r_step', type=int, default=1)
    parser.add_argument('-nb_samples', type=int, default=1)
    parser.add_argument('-timeout', type=int, default=60)
    parser.add_argument('-p', type=bool, default=False)
    parser.add_argument('-solver', type=str, default='bfs', choices=['bfs', 'dfs', 'a*'])
    parser.add_argument('-heuristic', type=str, default='manhattan', choices=['manhattan', 'deeplearning'])
    parser.add_argument('-model_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-action', type=str, default='solve', choices=['solve', 'plot'])
    parser.add_argument('-perf_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-append', type=bool, default=False)
    parser.add_argument('-nb_cpus', type=int, default=int(cpu_count() / 2))
    parser.add_argument('-shuffles_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-max_consecutive_timeout', type=int, default=0)
    parser = parser.parse_args()
    kw_args = {'n': parser.n,
               'm': parser.m,
               'max_consecutive_timeout': parser.max_consecutive_timeout}

    if parser.action in ['plot']:
            Solver.plot_performance(file_name()
                                    solver_name=None)

        ########################################################################################################################

        if '__main__' == __name__:
            main()




    solver = Solver.factory()
    r_min = parser.r_min
    r_max = parser.r_max
    r_step = parser.r_step
    if r_max is None:
        r_max = r_min
    perf_table = solver.performance(max_nb_shuffles=r_max,
                                    nb_samples=parser.nb_samples,
                                    time_out=parser.timeout,
                                    min_nb_shuffles=r_min,
                                    step_nb_shuffles=r_step,
                                    perfect_shuffle=parser.p,
                                    nb_cpus=parser.nb_cpus,
                                    shuffles_file_name=parser.shuffles_file_name)
    pprint(perf_table)
    if parser.save:
        if parser.append:
            try:
                perf_table = concat((read_pickle(parser.save), perf_table))
            except FileNotFoundError:
                pass
        subset = [Solver.solver_name_tag,
                  Solver.puzzle_type_tag,
                  Solver.puzzle_dimension_tag,
                  Solver.nb_shuffles_tag]
        perf_table = perf_table.drop_duplicates(subset=subset).sort_values(subset)
        to_pickle(perf_table, parser.save)
        pprint('Saved ', len(perf_table), ' rows of perf table to ', parser.save)
        pprint(perf_table)

########################################################################################################################

    
if '__main__' == __name__:
    if is_windows():
        PuzzleType = SlidingPuzzle
        dimension = (3, 3)
        perf_file_name = file_name(puzzle_type=PuzzleType,
                                   dimension=dimension,
                                   file_type='perf',
                                   name='perf')
        command_line_args = "%d -m=%d" % dimension
        command_line_args += " -r_min=0 -r_max=100 -r_step=10"
        command_line_args += " -nb_samples=250 -timeout=120 -nb_cpus=12 "
        command_line_args += "-perf_file_name=%s" % perf_file_name
        command_line_args += " -p=1"
        command_line_args += " -append=1"
        command_line_args += " -action=solve"
        #command_line_args += " -solver=bfs"
        command_line_args += " -max_consecutive_timeout=50"
        #command_line_args += " -solver=a* -heuristic=manhattan"
        layers = ('600', '300', '100')
        model_name = 'fully_connected_net_' + '_'.join(layers)
        model_file_name = file_name(puzzle_type=PuzzleType,
                                    dimension=dimension,
                                    file_type='models',
                                    name=model_name)
        command_line_args += " -solver=a* -heuristic=deeplearning -model_file_name=%s" % model_file_name
        command_line_args += " -shuffles_file_name=%s" % file_name(puzzle_type=PuzzleType,
                                                                   dimension=dimension,
                                                                   file_type='shuffles',
                                                                   name=model_name)
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
