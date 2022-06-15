########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from multiprocessing import cpu_count
from pandas import to_pickle, read_pickle, concat
from sys import argv
########################################################################################################################
from rubiks.heuristics.manhattan import Manhattan
from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.solvers.solver import Solver
from rubiks.solvers.bfssolver import BFSSolver
from rubiks.solvers.dfssolver import DFSSolver
from rubiks.solvers.astarsolver import AStarSolver
from rubiks.utils.utils import pprint, is_windows, g_not_a_pkl_file
########################################################################################################################


def main():
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-r_min', type=int, default=None)
    parser.add_argument('-r_max', type=int, default=None)
    parser.add_argument('-r_step', type=int, default=1)
    parser.add_argument('-s', type=int, default=1)
    parser.add_argument('-t', type=int, default=60)
    parser.add_argument('-p', type=bool, default=False)
    parser.add_argument('-solver', type=str, default='bfs', choices=['bfs', 'dfs', 'a*'])
    parser.add_argument('-heuristic', type=str, default='manhattan', choices=['manhattan', 'deeplearning'])
    parser.add_argument('-model', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-save', type=str, default=None)
    parser.add_argument('-append', type=bool, default=False)
    parser.add_argument('-nb_cpus', type=int, default=int(cpu_count() / 2))
    parser.add_argument('-shuffles_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('-max_consecutive_timeout', type=int, default=0)
    parser = parser.parse_args()
    kw_args = {'n': parser.n,
               'm': parser.m,
               'max_consecutive_timeout': parser.max_consecutive_timeout}
    if parser.solver == 'bfs':
        solver = BFSSolver
    elif parser.solver == 'dfs':
        solver = DFSSolver
    elif parser.solver == 'a*':
        solver = AStarSolver
    if solver is DFSSolver:
        kw_args.update({'limit': 100})
    elif solver is AStarSolver:
        if parser.heuristic == 'manhattan':
            heuristic = Manhattan
        elif parser.heuristic == 'deeplearning':
            kw_args.update({'model_file': parser.model})
            heuristic = DeepLearningHeuristic(**kw_args)
        else:
            raise NotImplementedError
        kw_args.update({'heuristic': heuristic})
    solver = solver(SlidingPuzzle, **kw_args)
    r_min = parser.r_min
    r_max = parser.r_max
    r_step = parser.r_step
    if r_max is None:
        r_max = r_min
    perf_table = solver.performance(max_nb_shuffles=r_max,
                                    nb_samples=parser.s,
                                    time_out=parser.t,
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
        command_line_args = "3 -m=3 -r_min=0 -r_max=100 -r_step=10 -s=250 -t=120 -nb_cpus=12 -save=C:/Users/franc/rubiks/perf/demo_8_puzzle_dl.pkl"
        command_line_args += " -p=1"
        command_line_args += " -append=1"
        #command_line_args += " -solver=bfs"
        command_line_args += " -max_consecutive_timeout=50"
        #command_line_args += " -solver=a* -heuristic=manhattan"
        command_line_args += " -solver=a* -heuristic=deeplearning -model=C:/Users/franc/rubiks/models/8_puzzle_fully_connected_net_1000_500_100.pkl"
        command_line_args += " -shuffles_file_name=C:/Users/franc/rubiks/shuffles/shuffles_data.pkl"
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
