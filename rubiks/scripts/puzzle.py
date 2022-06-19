########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from argparse import ArgumentParser
from math import inf
from sys import argv
from time import time as snap
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
from rubiks.utils.loggable import Loggable
from rubiks.utils.utils import is_windows, g_not_a_pkl_file, model_file_name, s_format
########################################################################################################################


def main():
    """ This script generates a random Sliding Puzzle and
    attempts to solve it using a solver of your choice """
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-m', type=int, default=None)
    parser.add_argument('-nb_shuffles', default=inf)
    parser.add_argument('-timeout', type=int, default=60)
    parser.add_argument('-puzzle_type', type=str, default=None,
                        choices=[Puzzle.sliding_puzzle, Puzzle.rubiks_cube])
    parser.add_argument('-solver_type', type=str, default='bfs', choices=['bfs', 'dfs', 'a*'])
    parser.add_argument('-heuristic_type', type=str, default='manhattan', choices=['manhattan', 'deeplearning'])
    parser.add_argument('-model_file_name', type=str, default=g_not_a_pkl_file)
    parser.add_argument('--log_solution', default=False, action='store_true')
    parser.add_argument('--check_optimal', default=False, action='store_true')
    parser = parser.parse_args()
    kw_args = {'n': parser.n,
               'm': parser.m,
               'puzzle_type': parser.puzzle_type,
               'solver_type': parser.solver_type,
               'heuristic_type': parser.heuristic_type,
               'model_file_name': parser.model_file_name}
    solver = Solver.factory(**kw_args)
    logger = Loggable(solver.name())
    puzzle = Puzzle.factory(**kw_args)
    logger.log_info('Shuffling %s %s time(s)' % (puzzle.name(), parser.nb_shuffles))
    puzzle = puzzle.apply_random_moves(parser.nb_shuffles, parser.nb_shuffles)
    logger.log_info(puzzle)
    try:
        b4 = snap()
        solution = solver.solve(puzzle, parser.timeout)
        info = 'Found solution of cost %s in %s' % (solution.cost, s_format(snap() - b4))
        if solver.know_to_be_optimal():
            info += '. This is an optimal solution.'
        logger.log_info(info)
        if parser.log_solution:
            logger.log_info(solution)
        if not solver.know_to_be_optimal() and parser.check_optimal:
            kw_args.update({'solver_type': 'a*',
                            'heuristic_type': 'manhattan'})
            optimal_solver = Solver.factory(**kw_args)
            b4 = snap()
            try:
                optimal_solution = optimal_solver.solve(puzzle, time_out=parser.timeout)
                if solution.cost != optimal_solution.cost:
                    logger.log_warning('Solution is not optimal!')
                    info = 'Optimal solution of cost %s in %s' % (optimal_solution.cost, s_format(snap() - b4))
                    logger.log_info(info)
                    if parser.log_solution:
                        logger.log_info('Optimal solution: ', optimal_solution)
                else:
                    logger.log_info('Solution is optimal')
            except TimeoutError:
                optimal_solver.log_error('Could not check optimality as timed out')
    except TimeoutError as error:
        logger.log_error(error)

########################################################################################################################


if '__main__' == __name__:
    puzzle_type = 'sliding_puzzle'
    dimension = (4, 3)
    nb_shuffles = inf
    solver_type = 'a*'
    heuristic_type = 'manhattan'
    timeout = 300
    log_solution = True
    check_optimal = True
    if is_windows():
        command_line_args = "%d -m=%d -nb_shuffles=%s" % (*dimension, nb_shuffles)
        command_line_args += " -puzzle_type=%s" % puzzle_type
        command_line_args += " -solver_type=%s" % solver_type
        command_line_args += " -timeout=%d" % timeout
        command_line_args += " -heuristic_type=%s" % heuristic_type
        layers = ('600', '300', '100')
        one_hot_encoding = False
        model_name = 'fully_connected_net_' + '_'.join(layers)
        if one_hot_encoding:
            model_name += '_one_hot_encoding'
            command_line_args += " --one_hot_encoding"
        command_line_args += " -model_file_name=%s" % (model_file_name(puzzle_type=puzzle_type,
                                                                       dimension=dimension,
                                                                       model_name=model_name))
        if log_solution:
            command_line_args += " --log_solution"
        if check_optimal:
            command_line_args += " --check_optimal"
        argv.extend(command_line_args.split(' '))
    main()

########################################################################################################################
