########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from functools import partial
from itertools import product
from math import inf
from matplotlib import pyplot as plt
from multiprocessing import Pool
from numpy import isnan
from pandas import concat, DataFrame, Series, read_pickle
from time import time as snap
########################################################################################################################
from rubiks.utils.utils import pprint
from rubiks.utils.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzled
########################################################################################################################


class Solver(Puzzled, Loggable, metaclass=ABCMeta):
    """ TBD """

    def __init__(self, puzzle_type, **kw_args):
        self.max_consecutive_timeout = kw_args.pop('max_consecutive_timeout', 0)
        Puzzled.__init__(self, puzzle_type, **kw_args)
        Loggable.__init__(self, self.name(), kw_args.pop('log_level', 'INFO'))

    @abstractmethod
    def solve_impl(self, puzzle, time_out, **kw_args):
        return

    def solve(self, nb_shuffles, time_out):
        """ A single puzzle to solve """
        start = snap()
        timed_out = False
        try:
            puzzle = self.puzzle_type.construct_puzzle(**self.kw_args)
            puzzle = puzzle.apply_random_moves(nb_shuffles)
            (cost, moves, expanded_nodes) = self.solve_impl(puzzle, time_out)
            run_time = snap() - start
            assert isinstance(cost, int)
            assert isinstance(moves, list)
            assert all(isinstance(move, self.puzzle_type.get_move_type()) for move in moves)
            assert isinstance(expanded_nodes, int)
        except Exception as error:
            if error is not RecursionError:
                self.log_error(error, '. nb_shuffles = ', nb_shuffles)
            run_time = time_out
            expanded_nodes = -1
            cost = 0
            timed_out = True
            moves = []
        return cost, moves, expanded_nodes, run_time, timed_out

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
    puzzle_type = 'puzzle_type'
    puzzle_dimension = 'puzzle_dimension'
    
    def performance(self,
                    max_nb_shuffles,
                    nb_samples,
                    time_out,
                    min_nb_shuffles=None,
                    step_nb_shuffles=1,
                    perfect_shuffle=False,
                    nb_cpus=1):
        """
        Runs the solver on a bunch of randomly generated puzzles (more or less shuffled from goal state)
        and returns statistics of the various attemps to solve them.
        params:
            max_nb_shuffles:
            nb_samples:
            time_out:
            min_nb_shuffles:
            step_nb_shuffles:
            perfect_shuffle:
            nb_cpus:
        returns:
            blablabla tbc
        """
        assert max_nb_shuffles > 0
        assert nb_samples > 0
        if min_nb_shuffles is None:
            min_nb_shuffles = 1
        assert min_nb_shuffles <= max_nb_shuffles
        dimension = str(tuple(self.puzzle_type.construct_puzzle(**self.kw_args).dimension()))
        cls = self.__class__
        res = {cls.solver_name: self.name(),
               cls.puzzle_type: self.puzzle_type.__name__,
               cls.puzzle_dimension: dimension,
               cls.nb_shuffle: [0],
               cls.nb_samples: [1],
               cls.avg_cost: [0],
               cls.max_cost: [0],
               cls.avg_expanded_nodes: [0],
               cls.max_expanded_nodes: [0],
               cls.nb_timeout: [0],
               cls.avg_run_time: [0],
               cls.max_run_time: [0],
               cls.pct_solved: [100]}
        performance = DataFrame(res)
        nan = float('nan')
        shuffles = list(range(min_nb_shuffles, max_nb_shuffles + 1, step_nb_shuffles))
        if perfect_shuffle:
            shuffles.append(inf)
        ''' @todo: parallelize the solving of diff shuffles 
        since I have so many cores on this machine, might as well '''
        for nb_shuffles in shuffles:
            if nb_shuffles <= 0:
                continue
            total_cost = 0
            max_cost = 0
            total_expanded_nodes = 0
            max_expanded_nodes = 0
            total_run_time = 0
            max_run_time = 0
            nb_timeout = 0
            res[cls.nb_shuffle] = nb_shuffles
            self.log_debug({'nb_shuffles': nb_shuffles, 'nb_cpus': nb_cpus})
            with Pool(nb_cpus) as pool:
                results = pool.map(partial(self.__class__.solve, self, nb_shuffles),
                                   [time_out]*(nb_samples if nb_shuffles > 0 else 1))
                consecutive_timeout = 0
                sample = 0
                for (cost, _, expanded_nodes, run_time, timed_out) in results:
                    if timed_out:
                        consecutive_timeout += 1
                        nb_timeout += 1
                    sample += 1
                    total_cost += cost
                    max_cost = max(max_cost, cost)
                    total_expanded_nodes += expanded_nodes
                    max_expanded_nodes = max(max_expanded_nodes, expanded_nodes)
                    total_run_time += run_time
                    max_run_time = max(max_run_time, run_time)
                    if self.max_consecutive_timeout and consecutive_timeout >= self.max_consecutive_timeout:
                        self.log_info('break out for nb_shuffles=', nb_shuffles,
                                      'as timed-out/error-ed %d times' % self.max_consecutive_timeout)
                        break
                div = nb_samples - nb_timeout
                if 0 == div:
                    div = float('nan')
                avg_cost = round(total_cost / div, 1)
                max_cost = max(max_cost, avg_cost)
                avg_expanded_nodes = round(total_expanded_nodes / div, 0)
                max_expanded_nodes = max(max_expanded_nodes, avg_expanded_nodes)
                avg_run_time = round(total_run_time / div, 3)
                max_run_time = max(max_run_time, avg_run_time)
                res[cls.nb_samples] = sample
                res[cls.avg_cost] = avg_cost
                res[cls.max_cost] = max_cost
                res[cls.avg_expanded_nodes] = avg_expanded_nodes
                res[cls.max_expanded_nodes] = max_expanded_nodes
                res[cls.nb_timeout] = nb_timeout
                res[cls.avg_run_time] = nan if isnan(avg_run_time) else int(avg_run_time * 1000)
                res[cls.max_run_time] = nan if isnan(max_run_time) else int(max_run_time * 1000)
                res[cls.pct_solved] = int(100 * (sample - nb_timeout) / nb_samples)
                performance = concat([performance,
                                      Series(res).to_frame().transpose()],
                                     ignore_index=True)
                self.log_info(performance)
        return performance

    def name(self):
        return '%s|%s' % (self.__class__.__name__,
                          self.puzzle_type.construct_puzzle(**self.kw_args).name())

    @staticmethod
    def plot_performance(performance_file_name,
                         solver_name=None,
                         puzzle_type=None,
                         puzzle_dimension=None):
        performance = read_pickle(performance_file_name)
        if solver_name:
            performance = performance[performance.solver_name.apply(lambda sn: sn.find(solver_name) >= 0)]
        if puzzle_type:
            performance = performance[performance.puzzle_type == puzzle_type]
        if puzzle_dimension:
            performance = performance[performance.puzzle_dimension == puzzle_dimension]
        shuffle_max = performance[Solver.nb_shuffle].replace(inf, -1).max() * 2
        performance.loc[:, Solver.nb_shuffle] = \
            performance[Solver.nb_shuffle].replace(inf, shuffle_max)
        pprint(performance)
        y = [Solver.avg_run_time, Solver.avg_cost, Solver.avg_expanded_nodes, Solver.pct_solved]
        n = int(len(y)/2)
        fig, axes = plt.subplots(2, n)
        gb = performance.groupby(Solver.solver_name)
        for r, c in product(range(2), range(n)):
            what = y[r * 2 + c]
            ax = axes[r, c]
            for sn, grp in gb:
                ax.scatter(x=Solver.nb_shuffle,
                           y=what,
                           data=grp,
                           label=sn)
            ax.title.set_text('%s vs %s' % (what, Solver.nb_shuffle))
            ax.set_xlabel(Solver.nb_shuffle)
            ax.set_ylabel(what)
            handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        plt.show()

########################################################################################################################
