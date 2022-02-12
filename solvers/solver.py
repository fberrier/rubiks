########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from pandas import DataFrame
from time import time as snap
########################################################################################################################
from rubiks.utils.loggable import Loggable
########################################################################################################################


class Solver(Loggable, metaclass=ABCMeta):
    """ TBD """

    def __init__(self, puzzle_type, **kw_args):
        self.puzzle_type = puzzle_type
        self.max_consecutive_timeout = kw_args.pop('max_consecutive_timeout', 10)
        self.kw_args = kw_args
        Loggable.__init__(self, self.name(), kw_args.pop('log_level', 'INFO'))

    def save(self, data_base):
        """ overwrite where meaningful """
        return

    @staticmethod
    def restore(data_base):
        """ overwrite where meaningful """
        return

    def learn(puzzle_type, **kw_args):
        """ overwrite this in the case where there is some learning to do """
        return

    @abstractmethod
    def solve_impl(self, puzzle, time_out):
        return

    def solve(self, nb_shuffles, time_out):
        """ A single puzzle to solve """
        puzzle = self.puzzle_type.construct_puzzle(**self.kw_args)
        puzzle = puzzle.apply_random_moves(nb_shuffles)
        (cost, moves, expanded_nodes) = self.solve_impl(puzzle, time_out)
        assert isinstance(cost, int)
        assert isinstance(moves, list)
        assert all(isinstance(move, self.puzzle_type.get_move_type()) for move in moves)
        assert isinstance(expanded_nodes, int)
        return cost, moves, expanded_nodes

    nb_shuffle = 'nb_shuffle'
    nb_samples = 'nb_samples'
    avg_cost = 'avg_cost'
    max_cost = 'max_cost'
    avg_expanded_nodes = 'avg_expanded_nodes'
    max_expanded_nodes = 'max_expanded_nodes'
    nb_timeout = 'nb_timeout'
    avg_run_time = 'avg_run_time (ms)'
    max_run_time = 'max_run_time (ms)'
    solver_name = 'solver_name'
    pct_solved = 'solved (%)'
    
    def performance(self, max_nb_shuffles, nb_samples, time_out):
        assert max_nb_shuffles > 0
        assert nb_samples > 0
        res = {self.__class__.solver_name: self.name(),
               self.__class__.nb_shuffle: [0],
               self.__class__.nb_samples: [1],
               self.__class__.avg_cost: [0],
               self.__class__.max_cost: [0],
               self.__class__.avg_expanded_nodes: [0],
               self.__class__.max_expanded_nodes: [0],
               self.__class__.nb_timeout: [0],
               self.__class__.avg_run_time: [0],
               self.__class__.max_run_time: [0],
               self.__class__.pct_solved: [100]}
        performance = DataFrame(res)
        for nb_shuffles in range(1, max_nb_shuffles + 1):
            total_cost = 0
            max_cost = 0
            total_expanded_nodes = 0
            max_expanded_nodes = 0
            total_run_time = 0
            max_run_time = 0
            nb_timeout = 0
            res[self.__class__.nb_shuffle] = nb_shuffles
            consecutive_timeout = 0
            self.log_debug('Calc performance for nb_shuffles=', nb_shuffles)
            for sample in range(nb_samples):
                run_time = -snap()
                try:
                    (cost, moves, expanded_nodes) = self.solve(nb_shuffles, time_out=time_out)
                    run_time += snap()
                    consecutive_timeout = 0
                except Exception as error:
                    self.log_debug(error)
                    run_time = time_out
                    nb_timeout += 1
                    cost = 0
                    consecutive_timeout += 1
                total_cost += cost
                max_cost = max(max_cost, cost)
                total_expanded_nodes += expanded_nodes
                max_expanded_nodes = max(max_expanded_nodes, expanded_nodes)
                total_run_time += run_time
                max_run_time = max(max_run_time, run_time)
                if consecutive_timeout > self.max_consecutive_timeout:
                    self.log_info('break out for nb_shuffles=', nb_shuffles,
                                  ' as timed-out %d times', self.max_consecutive_timeout)
                    break
            div = nb_samples + 1 - nb_timeout
            if 0 == div:
                div = float('nan')
            avg_cost = round(total_cost / div, 1)
            max_cost = max(max_cost, avg_cost)
            avg_expanded_nodes = round(total_expanded_nodes / div, 0)
            max_expanded_nodes = max(max_expanded_nodes, avg_expanded_nodes)
            avg_run_time = round(total_run_time / div, 3)
            max_run_time = max(max_run_time, avg_run_time)
            res[self.__class__.nb_samples] = sample + 1
            res[self.__class__.avg_cost] = avg_cost
            res[self.__class__.max_cost] = max_cost
            res[self.__class__.avg_expanded_nodes] = avg_expanded_nodes
            res[self.__class__.max_expanded_nodes] = max_expanded_nodes
            res[self.__class__.nb_timeout] = nb_timeout
            res[self.__class__.avg_run_time] = int(avg_run_time * 1000)
            res[self.__class__.max_run_time] = int(max_run_time * 1000)
            res[self.__class__.pct_solved] = int(100 * (sample + 1 - nb_timeout) / nb_samples)
            performance = performance.append(res, ignore_index=True)
        return performance

    def name(self):
        return '%s|%s' %(self.__class__.__name__,
                         self.puzzle_type.construct_puzzle(**self.kw_args).name())

 ########################################################################################################################
