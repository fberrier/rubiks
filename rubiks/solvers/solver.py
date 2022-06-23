########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from functools import partial
from itertools import product
from math import inf
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from matplotlib import ticker
from multiprocessing import Pool
from numpy import isnan, isinf
from pandas import concat, DataFrame, Series, read_pickle, to_pickle
from time import time as snap
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzled
from rubiks.solvers.solution import Solution
from rubiks.utils.loggable import Loggable
from rubiks.utils.utils import pprint, g_not_a_pkl_file, touch
########################################################################################################################


class Solver(Puzzled, Loggable, metaclass=ABCMeta):
    """ Base class for a puzzle solver. How it actually solves its puzzle type is
    left to derived classes implementations by overwriting the  'solve_impl' method
     """

    def __init__(self, puzzle_type, **kw_args):
        self.max_consecutive_timeout = kw_args.pop('max_consecutive_timeout', 0)
        Puzzled.__init__(self, puzzle_type, **kw_args)
        Loggable.__init__(self, log_level=kw_args.pop('log_level', 'INFO'))
        self.all_shuffles_data = None
        self.shuffles_data = None

    bfs_tag = 'bfs'
    breathfirst_tag = 'breathfirst'
    dfs_tag = 'dfs'
    depthfirst_tag = 'depthfirst'
    astar_tags = ['astar', 'a*']

    known_solver_types = [bfs_tag, breathfirst_tag,
                          dfs_tag, depthfirst_tag,
                          *astar_tags]

    @classmethod
    def factory(cls, solver_type, puzzle_type, **kw_args):
        solver_type = str(solver_type).lower()
        kw_args.update({'puzzle_type': puzzle_type})
        if any(solver_type.find(what) >= 0 for what in [cls.breathfirst_tag,
                                                        cls.bfs_tag]):
            from rubiks.solvers.bfssolver import BFSSolver as SolverType
        elif any(solver_type.find(what) >= 0 for what in [cls.dfs_tag,
                                                          cls.depthfirst_tag]):
            from rubiks.solvers.dfssolver import DFSSolver as SolverType
            kw_args.update({'limit': kw_args.get('limit', 100)})
        elif any(solver_type.find(what) >= 0 for what in cls.astar_tags):
            from rubiks.solvers.astarsolver import AStarSolver as SolverType
            kw_args.update({'heuristic_type': Heuristic.factory(**kw_args)})
        else:
            raise NotImplementedError('Unknown solver_type [%s]' % solver_type)
        return SolverType(**kw_args)

    @abstractmethod
    def know_to_be_optimal(self):
        """ Return True only if this is demonstrably returning optimal solutions """
        return False

    @abstractmethod
    def solve_impl(self, puzzle, time_out, **kw_args) -> Solution:
        return Solution(None, None, None)

    def solve(self, puzzle, time_out, **kw_args) -> Solution:
        return self.solve_impl(puzzle, time_out, **kw_args)

    def __job__(self, nb_shuffles, time_out, index=-1):
        """ A single puzzle to solve """
        start = snap()
        timed_out = False
        try:
            if self.shuffles_data and index >= 0:
                puzzle = self.shuffles_data[nb_shuffles][index][0]
            else:
                puzzle = self.get_goal().apply_random_moves(nb_moves=nb_shuffles,
                                                            min_no_loop=nb_shuffles)
            solution = self.solve_impl(puzzle, time_out)
            run_time = snap() - start
            assert isinstance(solution.cost, int)
            assert isinstance(solution.path, list)
            assert all(isinstance(move, self.get_puzzle_type().get_move_type()) for move in solution.path)
            assert isinstance(solution.expanded_nodes, int)
        except Exception as error:
            if error is not RecursionError:
                self.log_error(error, '. nb_shuffles = ', nb_shuffles, '. index=', index)
            run_time = time_out
            solution = Solution(0, [], -1, puzzle=puzzle)
            timed_out = True
        return solution.cost, solution.path, solution.expanded_nodes, run_time, timed_out, index

    nb_shuffles_tag = 'nb_shuffles'
    nb_samples_tag = 'nb_samples'
    avg_cost_tag = 'avg_cost'
    max_cost_tag = 'max_cost'
    avg_expanded_nodes_tag = 'avg_expanded_nodes'
    max_expanded_nodes_tag = 'max_expanded_nodes'
    nb_timeout_tag = 'nb_timeout'
    avg_run_time_tag = 'avg_run_time (ms)'
    max_run_time_tag = 'max_run_time (ms)'
    solver_name_tag = 'solver_name'
    pct_solved_tag = 'solved (%)'
    pct_optimal_tag = 'optimal (%)'
    puzzle_type_tag = 'puzzle_type'
    puzzle_dimension_tag = 'puzzle_dimension'
    
    def performance(self,
                    max_nb_shuffles,
                    nb_samples,
                    time_out,
                    min_nb_shuffles=None,
                    step_nb_shuffles=1,
                    perfect_shuffle=False,
                    nb_cpus=1,
                    shuffles_file_name=None):
        """
        Runs the solver on a bunch of randomly generated puzzles (more or less shuffled from goal state)
        and returns statistics of the various attempts to solve them.
        params:
            max_nb_shuffles:
            nb_samples:
            time_out:
            min_nb_shuffles:
            step_nb_shuffles:
            perfect_shuffle:
            nb_cpus:
            shuffles_file_name: if provided, can read puzzles' shuffles sequences from there, e.g. if
                                want several algorithms to use the same sequences for fairness
        returns:
            blablabla tbc
        """
        assert max_nb_shuffles > 0 or perfect_shuffle
        assert nb_samples > 0
        if min_nb_shuffles is None:
            min_nb_shuffles = 1
        assert min_nb_shuffles <= max_nb_shuffles
        dimension = tuple(self.puzzle_dimension())
        cls = self.__class__
        puzzle_type = self.get_puzzle_type().__name__
        res = {cls.solver_name_tag: self.name(),
               cls.puzzle_type_tag: puzzle_type,
               cls.puzzle_dimension_tag: str(dimension),
               cls.nb_shuffles_tag: [0],
               cls.nb_samples_tag: [1],
               cls.avg_cost_tag: [0],
               cls.max_cost_tag: [0],
               cls.avg_expanded_nodes_tag: [0],
               cls.max_expanded_nodes_tag: [0],
               cls.nb_timeout_tag: [0],
               cls.avg_run_time_tag: [0],
               cls.max_run_time_tag: [0],
               cls.pct_solved_tag: [100],
               cls.pct_optimal_tag: [100]}
        performance = DataFrame(res)
        nan = float('nan')
        shuffles = list(range(min_nb_shuffles, max_nb_shuffles + 1, step_nb_shuffles))
        if perfect_shuffle:
            shuffles.append(inf)
        if shuffles_file_name and shuffles_file_name != g_not_a_pkl_file:
            try:
                self.all_shuffles_data = read_pickle(shuffles_file_name)
            except FileNotFoundError:
                self.log_warning('Could not find shuffles_file_name \'%s\'' % shuffles_file_name)
                self.all_shuffles_data = dict()
                self.all_shuffles_data[puzzle_type] = dict()
                self.all_shuffles_data[puzzle_type][dimension] = dict()
            self.shuffles_data = self.all_shuffles_data
            if puzzle_type not in self.shuffles_data:
                self.shuffles_data[puzzle_type] = dict()
            self.shuffles_data = self.shuffles_data[puzzle_type]
            if dimension not in self.shuffles_data:
                self.shuffles_data[dimension] = dict()
            self.shuffles_data = self.shuffles_data[dimension]
            goal = self.get_goal()
            for nb_shuffles in shuffles:
                if nb_shuffles in self.shuffles_data:
                    required = max(0, nb_samples - len(self.shuffles_data[nb_shuffles]))
                else:
                    required = nb_samples
                    self.shuffles_data[nb_shuffles] = []
                new_shuffles = [(goal.apply_random_moves(nb_moves=nb_shuffles,
                                                         min_no_loop=nb_shuffles),
                                 inf) for r in range(required)]
                self.shuffles_data[nb_shuffles].extend(new_shuffles)
                if required > 0:
                    self.log_info('Saved %d more (overall %d) random puzzles for [%s, nb_shuffles=%s]' %
                                  (required,
                                   len(self.shuffles_data[nb_shuffles]),
                                   self.puzzle_name(),
                                   nb_shuffles))
        early_breakout = False
        pool_size = 1
        pool = Pool(pool_size)
        for nb_shuffles in shuffles:
            if nb_shuffles <= 0 or early_breakout:
                continue
            total_cost = 0
            max_cost = 0
            total_expanded_nodes = 0
            max_expanded_nodes = 0
            total_run_time = 0
            max_run_time = 0
            nb_timeout = 0
            res[cls.nb_shuffles_tag] = nb_shuffles
            self.log_debug({'nb_shuffles': nb_shuffles, 'nb_cpus': nb_cpus})
            sample_size = nb_samples if nb_shuffles > 0 else 1
            new_pool_size = min(nb_cpus, sample_size)
            if new_pool_size != pool_size:
                pool.close()
                pool.join()
                pool = Pool(new_pool_size)
                pool_size = new_pool_size
            results = pool.map(partial(self.__class__.__job__, self, nb_shuffles, time_out),
                               range(sample_size))
            consecutive_timeout = 0
            sample = 0
            nb_not_optimal = 0
            for (cost, moves, expanded_nodes, run_time, timed_out, index) in results:
                assert len(moves) == cost
                if timed_out:
                    consecutive_timeout += 1
                    nb_timeout += 1
                    self.log_debug('not optimal (timeout)')
                else:
                    if self.know_to_be_optimal():
                        if self.shuffles_data:
                            stored_cost = self.shuffles_data[nb_shuffles][index][1]
                            assert isinf(stored_cost) or stored_cost == cost
                            self.shuffles_data[nb_shuffles][index] = (self.shuffles_data[nb_shuffles][index][0],
                                                                      cost)
                        optimal_cost = cost
                        self.log_debug('Setting up optimal cost for nb_shuffles=',
                                       nb_shuffles,
                                       ' and index=',
                                       index,
                                       ': optimal cost=',
                                       optimal_cost)
                    if self.shuffles_data:
                        optimal_cost = self.shuffles_data[nb_shuffles][index][1]
                    if cost > optimal_cost:
                        nb_not_optimal += 1
                        self.log_debug('not optimal (cost=', cost, ' vs optimal=', optimal_cost, ')')
                    else:
                        self.log_debug('optimal')
                sample += 1
                total_cost += cost
                max_cost = max(max_cost, cost)
                total_expanded_nodes += expanded_nodes
                max_expanded_nodes = max(max_expanded_nodes, expanded_nodes)
                total_run_time += run_time
                max_run_time = max(max_run_time, run_time)
                if self.max_consecutive_timeout and consecutive_timeout >= self.max_consecutive_timeout:
                    self.log_debug('break out for nb_shuffles=', nb_shuffles,
                                   'as timed-out/error-ed %d times' % self.max_consecutive_timeout)
                    early_breakout = True
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
            res[cls.nb_samples_tag] = sample
            res[cls.avg_cost_tag] = avg_cost
            res[cls.max_cost_tag] = max_cost
            res[cls.avg_expanded_nodes_tag] = avg_expanded_nodes
            res[cls.max_expanded_nodes_tag] = max_expanded_nodes
            res[cls.nb_timeout_tag] = nb_timeout
            res[cls.avg_run_time_tag] = nan if isnan(avg_run_time) else int(avg_run_time * 1000)
            res[cls.max_run_time_tag] = nan if isnan(max_run_time) else int(max_run_time * 1000)
            res[cls.pct_solved_tag] = int(100 * (sample - nb_timeout) / nb_samples)
            res[cls.pct_optimal_tag] = int(100 * (sample - nb_not_optimal) / nb_samples)
            performance = concat([performance,
                                  Series(res).to_frame().transpose()],
                                 ignore_index=True)
            self.log_info(performance)
        if shuffles_file_name and shuffles_file_name != g_not_a_pkl_file:
            touch(shuffles_file_name)
            to_pickle(self.all_shuffles_data, shuffles_file_name)
            self.log_info('Saved all shuffles data to \'%s\'' % shuffles_file_name)
        pool.close()
        pool.join()
        return performance

    def name(self):
        return '%s[%s]' % (self.__class__.__name__, self.puzzle_name())

    def plot_performance(self,
                         performance_file_name,
                         solver_name=None,
                         puzzle_type=None,
                         puzzle_dimension=None):
        try:
            performance = read_pickle(performance_file_name)
        except FileNotFoundError:
            self.log_error('Cannot find \'%s\'. Did you really want to plot rather than solve?' % performance_file_name)
            return
        if solver_name:
            if not isinstance(solver_name, list):
                solver_name = [solver_name]

            def filter_sn(s_name):
                return any(s_name.lower().find(w.lower()) >= 0 for w in solver_name)
            performance = performance[performance.solver_name.apply(filter_sn)]
        if puzzle_type:
            performance = performance[performance.puzzle_type == puzzle_type]
        if puzzle_dimension:
            performance = performance[performance.puzzle_dimension == puzzle_dimension]
        print(performance[Solver.nb_shuffles_tag])
        assert inf in performance[Solver.nb_shuffles_tag].values, 'Fix code so it uses normal axes if not inf in there'
        shuffle_max = performance[Solver.nb_shuffles_tag].replace(inf, -1).max() * 2
        performance.loc[:, Solver.nb_shuffles_tag] = \
            performance[Solver.nb_shuffles_tag].replace(inf, shuffle_max)
        pprint(performance)
        y = [Solver.avg_run_time_tag,
             Solver.pct_optimal_tag,
             Solver.avg_expanded_nodes_tag,
             Solver.pct_solved_tag]
        n = int(len(y)/2)
        fig = plt.figure(performance_file_name)
        sps = GridSpec(n, 2, figure=fig)
        gb = performance.groupby(Solver.solver_name_tag)
        max_shuffle = max(performance[Solver.nb_shuffles_tag])
        for r, c in product(range(2), range(n)):
            what = y[r * 2 + c]
            bax = brokenaxes(xlims=((0, max_shuffle/2 + 1),
                                    (max_shuffle - 1.5, max_shuffle + 1.5)),
                             subplot_spec=sps[r, c])
            bax.set_title('%s vs %s' % (what, Solver.nb_shuffles_tag))
            if r == 1:
                bax.set_xlabel(Solver.nb_shuffles_tag)
            bax.set_ylabel(what)
            ticks = bax.get_xticks()
            labels = [['%d' % t for t in ticks[0]],
                      ['\u221e' for _ in ticks[1]]]
            bax.set_xticks('whatahorriblehack', 2, ticks, labels)
            for sn, grp in gb:
                bax.scatter(x=Solver.nb_shuffles_tag,
                            y=what,
                            data=grp,
                            label=sn)
            (handles, labels) = bax.get_legend_handles_labels()[0]
        fig.legend(handles, labels, loc='upper center')
        plt.show()

########################################################################################################################
