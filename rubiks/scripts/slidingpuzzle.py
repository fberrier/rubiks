########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from sys import argv
########################################################################################################################
from rubiks.puzzle.sliding import SlidingPuzzle
from rubiks.solvers.solver import Solver
from rubiks.utils.loggable import Loggable
from rubiks.utils.utils import is_windows, g_not_a_pkl_file, file_name
########################################################################################################################


def main():
    """ This script generates a random Sliding Puzzle and
    attempts to solve it using a solver of your choice """
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-nb_shuffles', type=int, default=None)
    parser.add_argument('-timeout', type=int, default=60)
    parser.add_argument('-solver', type=str, default='bfs', choices=['bfs', 'dfs', 'a*'])
    parser.add_argument('-heuristic', type=str, default='manhattan', choices=['manhattan', 'deeplearning'])
    parser.add_argument('-model_file_name', type=str, default=g_not_a_pkl_file)
    parser = parser.parse_args()
    kw_args = {'n': parser.n,
               'm': parser.m,
               'heuristic_type': parser.heuristic,
               'model_file_name': parser.model_file_name}
    solver = Solver.factory(solver_type=parser.solver,
                            puzzle_type=SlidingPuzzle,
                            **kw_args)
    logger = Loggable(solver.name())
    puzzle = SlidingPuzzle.construct_puzzle(**kw_args)
    logger.log_info('Shuffling %s %d time(s)' % (puzzle.name(), parser.nb_shuffles))
    puzzle = puzzle.apply_random_moves(parser.nb_shuffles, parser.nb_shuffles)
    logger.log_info(puzzle)
    try:
        solution = solver.solve(puzzle, parser.timeout)
        logger.log_info('Found solution of cost %s' % solution.cost)
        for move in solution.path:
            puzzle = puzzle.apply(move)
            logger.log_info(puzzle)
    except TimeoutError as error:
        logger.log_error(error)

########################################################################################################################


if '__main__' == __name__:
    PuzzleType = SlidingPuzzle
    dimension = (3, 3)
    if is_windows():
        command_line_args = "%d -m=%d -nb_shuffles=2" % dimension
        command_line_args += " -solver=a*"
        command_line_args += " -heuristic=deeplearning"
        command_line_args += " -model_file_name=%s" % (file_name(puzzle_type=PuzzleType,
                                                                 dimension=dimension,
                                                                 file_type='models',
                                                                 name='fully_connected'))
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
