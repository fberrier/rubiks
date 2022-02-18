########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
########################################################################################################################
from rubiks.heuristics.manhattan import Manhattan
from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.solvers.bfssolver import BFSSolver
from rubiks.solvers.dfssolver import DFSSolver
from rubiks.solvers.astarsolver import AStarSolver
from rubiks.utils.utils import pprint
########################################################################################################################


def main():
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-rmin', type=int, default=None)
    parser.add_argument('-rmax', type=int, default=None)
    parser.add_argument('-s', type=int, default=1)
    parser.add_argument('-t', type=int, default=60)
    parser.add_argument('-solver', type=str, default='bfs', choices=['bfs', 'dfs', 'a*'])
    parser.add_argument('-heuristic', type=str, default='manhattan', choices=['manhattan', 'deeplearning'])
    parser.add_argument('-data_base', type=str, default='')
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
            kw_args.update({'data_base': parser.data_base})
            heuristic = DeepLearningHeuristic(**kw_args)
        else:
            raise NotImplementedError
        kw_args.update({'heuristic': heuristic})
    solver = solver(SlidingPuzzle, **kw_args)
    rmin = parser.rmin
    rmax = parser.rmax
    if rmax is None:
        rmax = rmin
    pprint(solver.performance(max_nb_shuffles=rmax,
                              nb_samples=parser.s,
                              time_out=parser.t,
                              min_nb_shuffles=rmin))

########################################################################################################################

    
if '__main__' == __name__:
    main()

########################################################################################################################
