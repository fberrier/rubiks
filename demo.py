########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from pandas import to_pickle
from sys import argv, platform
########################################################################################################################
from rubiks.heuristics.manhattan import Manhattan
from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.solvers.bfssolver import BFSSolver
from rubiks.solvers.dfssolver import DFSSolver
from rubiks.solvers.astarsolver import AStarSolver
from rubiks.utils.utils import pprint, is_windows
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
    parser.add_argument('-solver', type=str, default='bfs', choices=['bfs', 'dfs', 'a*'])
    parser.add_argument('-heuristic', type=str, default='manhattan', choices=['manhattan', 'deeplearning'])
    parser.add_argument('-model', type=str, default='not_a_file.pkl')
    parser.add_argument('-save', type=str, default=None)
    parser = parser.parse_args()
    kw_args = {'n': parser.n,
               'm': parser.m}
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
                                    step_nb_shuffles=r_step)
    pprint(perf_table)
    if parser.save:
        to_pickle(perf_table, parser.save)

########################################################################################################################

    
if '__main__' == __name__:
    if is_windows():
        command_line_args = "2 -m=2 -r_min=1 -r_max=10 -s=100"
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
