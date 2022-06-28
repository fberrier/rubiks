########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from functools import partial
from itertools import product
from math import inf
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from multiprocessing import Pool
from numpy import isnan, isinf
from pandas import concat, DataFrame, Series, read_pickle
from time import time as snap
########################################################################################################################
from rubiks.thridparties import brokenaxes
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.core.factory import Factory
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.puzzled import Puzzled
from rubiks.search.searchstrategy import SearchStrategy
from rubiks.solvers.solution import Solution
from rubiks.utils.utils import pprint, to_pickle, remove_file, s_format
########################################################################################################################


class Solver(Factory, Puzzled, Loggable, metaclass=ABCMeta):
    """ Base class for a puzzle solver. How it actually solves its puzzle type is
    left to derived classes implementations by overwriting the  'solve_impl' method
     """

    solver_type = 'solver_type'
    bfs = SearchStrategy.bfs
    dfs = SearchStrategy.dfs
    astar = SearchStrategy.astar
    known_solver_types = [bfs, dfs, astar]
    time_out = 'time_out'
    max_consecutive_timeout = 'max_consecutive_timeout'
    default_max_consecutive_timeout = 0
    log_solution = 'log_solution'
    check_optimal = 'check_optimal'

    def __init__(self, **kw_args):
        Factory.__init__(self, **kw_args)
        Puzzled.__init__(self, **kw_args)
        Loggable.__init__(self, **kw_args)
        self.all_shuffles_data = None
        self.shuffles_data = None

    @classmethod
    def factory_key_name(cls):
        return cls.solver_type

    @classmethod
    def widget_types(cls):
        from rubiks.solvers.bfssolver import BFSSolver
        from rubiks.solvers.dfssolver import DFSSolver
        from rubiks.solvers.astarsolver import AStarSolver
        return {cls.bfs: BFSSolver,
                cls.dfs: DFSSolver,
                cls.astar: AStarSolver}

    @classmethod
    def additional_dependencies(cls):
        return Heuristic.get_widgets() + [Heuristic]

    @abstractmethod
    def know_to_be_optimal(self):
        """ Return True only if this is demonstrably returning optimal solutions """
        return False

    def solve_impl(self, puzzle, **kw_args) -> Solution:
        """ Can over-write if need to"""
        return Solution.failure(puzzle)

    def solve(self, puzzle, **kw_args) -> Solution:
        kw_args = {**self.get_config(), **kw_args}
        try:
            solution = self.solve_impl(puzzle, **kw_args)
        except Exception as error:
            solution = Solution.failure(puzzle=puzzle,
                                        solver_name=self.get_name(),
                                        failure_reason=error)
        if self.log_solution:
            self.log_info(solution)
        if self.check_optimal:
            if self.know_to_be_optimal() and not solution.failed():
                self.log_info('Solution is optimal')
            else:
                assert self.get_puzzle_type() == Puzzle.sliding_puzzle, \
                    'No admissible heuristic for %s' % self.get_puzzle_type()
                kw_args.update({Solver.solver_type: Solver.astar,
                                Heuristic.heuristic_type: Heuristic.manhattan,
                                Solver.log_solution: False,
                                Solver.check_optimal: False})
                optimal_solver = Solver.factory(**kw_args)
                b4 = snap()
                try:
                    optimal_solution = optimal_solver.solve(puzzle, **kw_args)
                    if solution.cost != optimal_solution.cost:
                        self.log_warning('Solution is not optimal!')
                        info = 'Optimal solution of cost %s in %s' % (optimal_solution.cost,
                                                                      s_format(snap() - b4))
                        self.log_info(info)
                        if self.log_solution:
                            self.log_info('Optimal solution: ', optimal_solution)
                    else:
                        self.log_info('Solution is optimal')
                except TimeoutError:
                    optimal_solver.log_error('Could not check optimality as timed out')
        return solution

    def __job__(self, nb_shuffles, index=-1):
        """ A single puzzle to solve """
        start = snap()
        timed_out = False
        try:
            if self.shuffles_data and index >= 0:
                puzzle = self.shuffles_data[nb_shuffles][index][0]
            else:
                puzzle = self.get_goal().apply_random_moves(nb_moves=nb_shuffles,
                                                            min_no_loop=nb_shuffles)
            solution = self.solve_impl(puzzle)
            run_time = snap() - start
            assert isinstance(solution.cost, int)
            assert isinstance(solution.path, list)
            assert all(isinstance(move, self.get_puzzle_type().get_move_type()) for move in solution.path)
            assert isinstance(solution.expanded_nodes, int)
        except Exception as error:
            if error is not RecursionError:
                self.log_error(error, '. nb_shuffles = ', nb_shuffles, '. index=', index)
            run_time = float(self.time_out)
            solution = Solution.failure(puzzle)
            timed_out = True
        return solution.cost, solution.path, solution.expanded_nodes, run_time, timed_out, index

    nb_shuffles = 'nb_shuffles'
    min_nb_shuffles = 'min_nb_shuffles'
    max_nb_shuffles = 'max_nb_shuffles'
    step_nb_shuffles = 'step_nb_shuffles'
    add_perfect_shuffle = 'add_perfect_shuffle'
    nb_samples = 'nb_samples'
    nb_cpus = 'nb_cpus'
    append = 'append'
    performance_file_name = 'performance_file_name'
    shuffles_file_name = 'shuffles_file_name'
    avg_cost = 'avg_cost'
    max_cost = 'max_cost'
    avg_expanded_nodes = 'avg_expanded_nodes'
    max_expanded_nodes = 'max_expanded_nodes'
    nb_timeout = 'nb_timeout'
    avg_run_time = 'avg_run_time (ms)'
    max_run_time = 'max_run_time (ms)'
    solver_name = 'solver_name'
    pct_solved = 'solved (%)'
    pct_optimal = 'optimal (%)'
    puzzle_type = 'puzzle_type'
    puzzle_dimension = 'puzzle_dimension'
    action_type = 'action_type'
    do_plot = 'do_plot'
    do_solve = 'do_solve'
    do_performance_test = 'do_performance_test'
    do_cleanup_performance_file = 'do_cleanup_performance_file'
    do_cleanup_shuffles_file = 'do_cleanup_shuffles_file'
    known_action_type = [do_solve,
                         do_plot,
                         do_performance_test,
                         do_cleanup_performance_file,
                         do_cleanup_shuffles_file,
                         ]

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.time_out,
                         type=int,
                         default=0)
        cls.add_argument(parser,
                         type=int,
                         field=cls.max_consecutive_timeout,
                         default=cls.default_max_consecutive_timeout)
        cls.add_argument(parser,
                         field=cls.solver_type,
                         choices=cls.known_solver_types,
                         default=cls.astar)
        cls.add_argument(parser,
                         cls.nb_shuffles,
                         type=float,
                         default=None)
        cls.add_argument(parser,
                         cls.min_nb_shuffles,
                         type=int,
                         default=None)
        cls.add_argument(parser,
                         cls.max_nb_shuffles,
                         type=int,
                         default=None)
        cls.add_argument(parser,
                         cls.step_nb_shuffles,
                         type=int,
                         default=1)
        cls.add_argument(parser,
                         cls.nb_samples,
                         type=int,
                         default=1000)
        cls.add_argument(parser,
                         cls.nb_cpus,
                         type=int,
                         default=1)
        cls.add_argument(parser,
                         cls.append,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         cls.log_solution,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         cls.check_optimal,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         cls.add_perfect_shuffle,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         cls.performance_file_name,
                         type=str,
                         default=None)
        cls.add_argument(parser,
                         cls.shuffles_file_name,
                         type=str,
                         default=None)
        cls.add_argument(parser,
                         cls.action_type,
                         type=str,
                         default=None,
                         choices=cls.known_action_type)
    
    def performance_test(self):
        """
        Runs the solver on a bunch of randomly generated puzzles (more or less shuffled from goal state)
        and returns statistics of the various attempts to solve them.
        """
        assert self.max_nb_shuffles > 0 or self.add_perfect_shuffle, \
            'Cannot run performance_test with negative max_nb_shuffles and not add_perfect_shuffle'
        assert self.nb_samples > 0, 'Cannot run performance_test with nb_samples <= 0'
        if self.min_nb_shuffles is None:
            self.min_nb_shuffles = 1
        assert self.min_nb_shuffles <= self.max_nb_shuffles
        dimension = tuple(self.get_puzzle_dimension())
        cls = self.__class__
        puzzle_type = self.get_puzzle_type().__name__
        res = {cls.solver_name: self.name(),
               cls.puzzle_type: puzzle_type,
               cls.puzzle_dimension: str(dimension),
               cls.nb_shuffles: [0],
               cls.nb_samples: [1],
               cls.avg_cost: [0],
               cls.max_cost: [0],
               cls.avg_expanded_nodes: [0],
               cls.max_expanded_nodes: [0],
               cls.nb_timeout: [0],
               cls.avg_run_time: [0],
               cls.max_run_time: [0],
               cls.pct_solved: [100],
               cls.pct_optimal: [100]}
        performance = DataFrame(res)
        nan = float('nan')
        shuffles = list(range(self.min_nb_shuffles,
                              self.max_nb_shuffles + 1,
                              self.step_nb_shuffles))
        if self.add_perfect_shuffle:
            shuffles.append(inf)
        if self.shuffles_file_name:
            try:
                self.all_shuffles_data = read_pickle(self.shuffles_file_name)
            except FileNotFoundError:
                self.log_warning('Could not find shuffles_file_name \'%s\'' % self.shuffles_file_name)
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
                    required = max(0, self.nb_samples - len(self.shuffles_data[nb_shuffles]))
                else:
                    required = self.nb_samples
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
            res[cls.nb_shuffles] = nb_shuffles
            sample_size = self.nb_samples if nb_shuffles > 0 else 1
            new_pool_size = min(self.nb_cpus, sample_size)
            if new_pool_size != pool_size:
                pool.close()
                pool.join()
                pool = Pool(new_pool_size)
                pool_size = new_pool_size
            results = pool.map(partial(cls.__job__, self, nb_shuffles),
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
            div = self.nb_samples - nb_timeout
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
            res[cls.pct_solved] = int(100 * (sample - nb_timeout) / self.nb_samples)
            res[cls.pct_optimal] = int(100 * (sample - nb_not_optimal) / self.nb_samples)
            performance = concat([performance,
                                  Series(res).to_frame().transpose()],
                                 ignore_index=True)
            self.log_info(performance)
        if self.shuffles_file_name:
            to_pickle(self.all_shuffles_data, self.shuffles_file_name)
            self.log_info('Saved all shuffles data to \'%s\'' % self.shuffles_file_name)
        pool.close()
        pool.join()
        self.log_info(performance)
        if self.append and self.performance_file_name:
            try:
                performance = concat((read_pickle(self.performance_file_name), performance))
            except FileNotFoundError:
                pass
        subset = [cls.solver_name,
                  cls.puzzle_type,
                  cls.puzzle_dimension,
                  cls.nb_shuffles]
        performance = performance.drop_duplicates(subset=subset).sort_values(subset)
        if self.performance_file_name:
            to_pickle(performance, self.performance_file_name)
            self.log_info('Saved ',
                          len(performance),
                          ' rows of perf table to \'%s\'' % self.performance_file_name)
        self.log_info(performance)

    def get_name(self):
        return '%s[%s]' % (self.__class__.__name__, self.puzzle_name())

    def action(self):
        config = self.get_config()
        if self.do_plot == self.action_type:
            self.plot_performance()
        elif self.do_solve == self.action_type:
            puzzle = Puzzle.factory(**self.get_config())
            puzzle = puzzle.apply_random_moves(nb_moves=config[self.__class__.nb_shuffles])
            return self.solve(puzzle)
        elif self.do_performance_test == self.action_type:
            self.performance_test()
        elif self.do_cleanup_performance_file == self.action_type:
            self.cleanup_perf_file()
        elif self.do_cleanup_shuffles_file == self.action_type:
            self.cleanup_shuffles_file()
        else:
            raise NotImplementedError('Unknown action_type [%s]' % self.action_type)

    def cleanup_perf_file(self):
        try:
            remove_file(self.performance_file_name)
            self.log_info('Removed \'%s\'' % self.performance_file_name)
        except FileNotFoundError:
            pass

    def cleanup_shuffles_file(self):
        try:
            remove_file(self.shuffles_file_name)
            self.log_info('Removed \'%s\'' % self.shuffles_file_name)
        except FileNotFoundError:
            pass

    def plot_performance(self):
        try:
            performance = read_pickle(self.performance_file_name)
        except FileNotFoundError:
            self.log_error('Cannot find \'%s\'. Did you really want to plot rather than solve?' %
                           self.performance_file_name)
            return
        self.log_info(performance)
        assert 1 == len(set(performance.puzzle_type))
        assert 1 == len(set(performance.puzzle_dimension))
        assert inf in performance[Solver.nb_shuffles].values, 'Fix code so it uses normal axes if not inf in there'
        shuffle_max = performance[Solver.nb_shuffles].replace(inf, -1).max() * 2
        performance.loc[:, Solver.nb_shuffles] = \
            performance[Solver.nb_shuffles].replace(inf, shuffle_max)
        pprint(performance)
        y = [Solver.avg_run_time,
             Solver.pct_optimal,
             Solver.avg_expanded_nodes,
             Solver.pct_solved]
        n = int(len(y)/2)
        fig = plt.figure(self.performance_file_name)
        sps = GridSpec(n, 2, figure=fig)
        gb = performance.groupby(Solver.solver_name)
        max_shuffle = max(performance[Solver.nb_shuffles])
        for r, c in product(range(2), range(n)):
            what = y[r * 2 + c]
            bax = brokenaxes(xlims=((0, max_shuffle/2 + 1),
                                    (max_shuffle - 1.5, max_shuffle + 1.5)),
                             subplot_spec=sps[r, c])
            bax.set_title('%s vs %s' % (what, Solver.nb_shuffles))
            if r == 1:
                bax.set_xlabel(Solver.nb_shuffles)
            bax.set_ylabel(what)
            ticks = bax.get_xticks()
            labels = [['%d' % t for t in ticks[0]],
                      ['\u221e' for _ in ticks[1]]]
            bax.set_xticks('whatahorriblehack', 2, ticks, labels)
            for sn, grp in gb:
                bax.scatter(x=Solver.nb_shuffles,
                            y=what,
                            data=grp,
                            label=sn)
            (handles, labels) = bax.get_legend_handles_labels()[0]
        fig.legend(handles, labels, loc='upper center')
        plt.show()

########################################################################################################################
