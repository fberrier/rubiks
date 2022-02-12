########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from pandas import DataFrame
from time import time as snap
########################################################################################################################


class Solver(metaclass=ABCMeta):
    """ TBD """

    def __init__(self, puzzle_type, **kw_args):
        self.puzzle_type = puzzle_type
        self.max_consecutive_timeout = kw_args.pop('max_consecutive_timeout', 10)
        self.kw_args = kw_args

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
        (cost, moves) = self.solve_impl(puzzle, time_out)
        assert isinstance(cost, int)
        assert isinstance(moves, list)
        assert all(isinstance(move, self.puzzle_type.get_move_type()) for move in moves)
        return cost, moves

    nb_shuffle = 'nb_shuffle'
    nb_samples = 'nb_samples'
    avg_cost = 'avg_cost'
    max_cost = 'max_cost'
    nb_timeout = 'nb_timeout'
    avg_run_time = 'avg_run_time'
    max_run_time = 'max_run_time'
    solver_name = 'solver_name'
    
    def performance(self, max_nb_shuffles, nb_samples, time_out):
        assert max_nb_shuffles > 0
        assert nb_samples > 0
        res = {__class__.solver_name: self.name(),
               __class__.nb_shuffle: [0],
               __class__.nb_samples: [1],
               __class__.avg_cost: [0],
               __class__.max_cost: [0],
               __class__.nb_timeout: [0],
               __class__.avg_run_time: [0],
               __class__.max_run_time: [0]}
        performance = DataFrame(res)
        for nb_shuffles in range(1, max_nb_shuffles + 1):
            total_cost = 0
            max_cost = 0
            total_run_time = 0
            max_run_time = 0
            nb_timeout = 0
            res[__class__.nb_shuffle] = nb_shuffles
            consecutive_timeout = 0
            for sample in range(nb_samples):
                run_time = -snap()
                try:
                    (cost, moves) = self.solve(nb_shuffles, time_out=time_out)
                    run_time += snap()
                    consecutive_timeout = 0
                except Exception as error:
                    print('Error solving puzzle: ', error)
                    run_time = time_out
                    nb_timeout += 1
                    cost = 0
                    consecutive_timeout += 1
                total_cost += cost
                max_cost = max(max_cost, cost)
                total_run_time += run_time
                max_run_time = max(max_run_time, run_time)
                if consecutive_timeout > self.max_consecutive_timeout:
                    break
            div = nb_samples + 1 - nb_timeout
            if 0 == div:
                div = float('nan')
            avg_cost = total_cost / div
            max_cost = max(max_cost, avg_cost)
            avg_run_time = total_run_time / div
            max_run_time = max(max_run_time, avg_run_time)
            res[__class__.nb_samples] = sample + 1
            res[__class__.avg_cost] = avg_cost
            res[__class__.max_cost] = max_cost
            res[__class__.nb_timeout] = nb_timeout
            res[__class__.avg_run_time] = int(avg_run_time * 1000)
            res[__class__.max_run_time] = int(max_run_time * 1000)
            performance = performance.append(res, ignore_index=True)
        return performance

    def name(self):
        return '%s|%s' %(__class__.__name__,
                         self.puzzle_type.construct_puzzle(**self.kw_args).name())

 ########################################################################################################################
