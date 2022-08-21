########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from functools import partial
from itertools import product, cycle
from math import inf, ceil
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from multiprocessing import Pool
from numpy import isnan, isinf, median
from pandas import concat, DataFrame, Series, read_pickle
from time import time as snap
########################################################################################################################
from rubiks.thirdparties.brokenaxes import brokenaxes
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.core.factory import Factory
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.puzzled import Puzzled
from rubiks.search.searchstrategy import SearchStrategy
from rubiks.solvers.solution import Solution
from rubiks.utils.utils import pprint, to_pickle, remove_file, s_format, \
    pformat, ms_format, get_model_file_name, number_format
########################################################################################################################


class Solver(Factory, Puzzled, Loggable, metaclass=ABCMeta):
    """ Base class for a puzzle solver. How it actually solves its puzzle type is
    left to derived classes implementations by overwriting the  'solve_impl' method
     """

    solver_type = 'solver_type'
    bfs = SearchStrategy.bfs
    dfs = SearchStrategy.dfs
    astar = SearchStrategy.astar
    mcts = 'mcts'
    naive = 'naive'
    kociemba = 'kociemba'
    known_solver_types = [bfs, dfs, astar]
    time_out = 'time_out'
    max_consecutive_timeout = 'max_consecutive_timeout'
    default_max_consecutive_timeout = 0
    log_solution = 'log_solution'
    check_optimal = 'check_optimal'
    do_not_reattempt_failed = 'do_not_reattempt_failed'

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
        from rubiks.solvers.astarsolver import AStarSolver
        from rubiks.solvers.bfssolver import BFSSolver
        from rubiks.solvers.dfssolver import DFSSolver
        from rubiks.solvers.kociembasolver import KociembaSolver
        from rubiks.solvers.mctssolver import MonteCarloTreeSearchSolver
        from rubiks.solvers.naiveslidingsolver import NaiveSlidingSolver
        return {cls.astar: AStarSolver,
                cls.bfs: BFSSolver,
                cls.dfs: DFSSolver,
                cls.kociemba: KociembaSolver,
                cls.mcts: MonteCarloTreeSearchSolver,
                cls.naive: NaiveSlidingSolver,}

    @classmethod
    def additional_dependencies(cls):
        return Heuristic.get_widgets() + [Heuristic]

    @abstractmethod
    def known_to_be_optimal(self):
        """ Return True only if this is demonstrably returning optimal solutions """
        return False

    def solve_impl(self, puzzle, **kw_args) -> Solution:
        """ Can over-write if need to"""
        return Solution.failure(puzzle)

    def solve(self, puzzle, **kw_args) -> Solution:
        kw_args = {**self.get_config(), **kw_args}
        try:
            b4 = snap()
            solution = self.solve_impl(puzzle, **kw_args)
            run_time = snap() - b4
            solution.set_run_time(run_time)
            solution.add_additional_info(run_time=ms_format(run_time))
        except Exception as error:
            solution = Solution.failure(puzzle=puzzle,
                                        solver_name=self.get_name(),
                                        failure_reason=error)
        if self.log_solution:
            self.log_info(solution)
        if self.check_optimal:
            if self.known_to_be_optimal() and not solution.failed():
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

    def __job__(self, nb_shuffles, index=-1) -> Solution:
        """ A single puzzle to solve """
        start = snap()
        try:
            if self.shuffles_data and index >= 0:
                puzzle = self.shuffles_data[nb_shuffles][index][0]
                if 3 <= len(self.shuffles_data[nb_shuffles][index]) and \
                        self.get_name() in self.shuffles_data[nb_shuffles][index][2]:
                    failed = self.shuffles_data[nb_shuffles][index][2][self.get_name()].failed()
                    if self.do_not_reattempt_failed or not failed:
                        if self.verbose:
                            print('Already ',
                                  (' failed ' if failed else 'solved '),
                                  puzzle, ' # ', index + 1, ' with ', self.get_name())
                        return self.shuffles_data[nb_shuffles][index][2][self.get_name()]
                    if failed:
                        if self.verbose:
                            print('Will reattempt failed ',
                                  puzzle, ' # ', index + 1, ' with ', self.get_name())
            else:
                puzzle = self.get_goal().apply_random_moves(nb_moves=nb_shuffles,
                                                            min_no_loop=nb_shuffles)
            if (index + 1) in self.skip:
                if self.verbose:
                    print('Skipping ', puzzle, ' # ', index + 1, ' ... ')
                solution = Solution.failure(puzzle)
                solution.set_run_time(self.time_out)
                solution.set_additional_info(index=index)
                return solution
            if self.verbose:
                print('Starting solving ', puzzle, ' # ', index + 1, ' ... ')
            solution = self.solve_impl(puzzle, **self.get_config())
            run_time = snap() - start
            solution.set_run_time(run_time)
            assert isinf(solution.cost) or isinstance(solution.cost, int)
            assert isinstance(solution.path, list)
            assert all(isinstance(move, self.get_goal().get_move_type()) for move in solution.path)
            assert isnan(solution.expanded_nodes) or isinstance(solution.expanded_nodes, int)
            if self.verbose:
                if solution.failed():
                    print(' ... failed to solve ', puzzle, ' # ', index + 1)
                else:
                    print(' ... solved ',
                          puzzle,
                          ' # ',
                          index + 1,
                          ' with cost ',
                          solution.cost,
                          ' in ',
                          s_format(run_time))
        except Exception as error:
            solution = Solution.failure(puzzle)
            solution.set_run_time(self.time_out)
            if self.verbose:
                print('Failed to solve ', puzzle, ': ', error)
        solution.set_additional_info(index=index)
        return solution

    nb_shuffles = 'nb_shuffles'
    min_nb_shuffles = 'min_nb_shuffles'
    max_nb_shuffles = 'max_nb_shuffles'
    step_nb_shuffles = 'step_nb_shuffles'
    add_perfect_shuffle = 'add_perfect_shuffle'
    nb_samples = 'nb_samples'
    chunk_size = 'chunk_size'
    nb_cpus = 'nb_cpus'
    append = 'append'
    performance_file_name = 'performance_file_name'
    shuffles_file_name = 'shuffles_file_name'
    avg_cost = 'avg_cost'
    median_cost = 'median_cost'
    max_cost = 'max_cost'
    avg_expanded_nodes = 'avg_expanded_nodes'
    median_expanded_nodes = 'median_expanded_nodes'
    max_expanded_nodes = 'max_expanded_nodes'
    nb_timeout = 'nb_timeout'
    avg_run_time = 'avg_run_time (ms)'
    median_run_time = 'median_run_time (ms)'
    max_run_time = 'max_run_time (ms)'
    solver_name = 'solver_name'
    pct_solved = 'solved (%)'
    pct_optimal = 'optimal (%)'
    optimality_score = 'optimality_score (%)'
    puzzle_type = 'puzzle_type'
    puzzle_dimension = 'puzzle_dimension'
    action_type = 'action_type'
    do_plot = 'do_plot'
    do_solve = 'do_solve'
    skip = 'skip'
    do_performance_test = 'do_performance_test'
    do_cleanup_performance_file = 'do_cleanup_performance_file'
    do_cleanup_shuffles_file = 'do_cleanup_shuffles_file'
    known_action_type = [do_solve,
                         do_plot,
                         do_performance_test,
                         do_cleanup_performance_file,
                         do_cleanup_shuffles_file,
                         ]
    performance_metrics = 'performance_metrics'
    default_performance_metrics = [median_run_time,
                                   pct_optimal,
                                   median_expanded_nodes,
                                   pct_solved]
    verbose = 'verbose'
    fig_size = 'fig_size'
    loc = 'loc'

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.performance_metrics,
                         type=str,
                         nargs='+',
                         default=cls.default_performance_metrics)
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
                         cls.chunk_size,
                         type=int,
                         default=0)
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
        cls.add_argument(parser,
                         cls.fig_size,
                         type=int,
                         nargs='+',
                         default=[16, 12])
        cls.add_argument(parser,
                         cls.verbose,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         cls.loc,
                         type=str,
                         default='best')
        cls.add_argument(parser,
                         cls.do_not_reattempt_failed,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.skip,
                         type=int,
                         nargs='+')
    
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
        puzzle_type = self.get_puzzle_type()
        res = {cls.solver_name: self.get_name(),
               cls.puzzle_type: puzzle_type,
               cls.puzzle_dimension: str(dimension),
               cls.nb_shuffles: [0],
               cls.nb_samples: [1],
               cls.avg_cost: [0],
               cls.median_cost: [0],
               cls.max_cost: [0],
               cls.avg_expanded_nodes: [0],
               cls.median_expanded_nodes: [0],
               cls.max_expanded_nodes: [0],
               cls.nb_timeout: [0],
               cls.avg_run_time: [0],
               cls.median_run_time: [0],
               cls.max_run_time: [0],
               cls.pct_solved: [100],
               cls.pct_optimal: [100],
               cls.optimality_score: [100],
               }
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
            for nb_shuffles in shuffles:
                if nb_shuffles in self.shuffles_data:
                    required = max(0, self.nb_samples - len(self.shuffles_data[nb_shuffles]))
                else:
                    required = self.nb_samples
                    self.shuffles_data[nb_shuffles] = list()
                new_shuffles = [(self.get_goal().apply_random_moves(nb_moves=nb_shuffles,
                                                                    min_no_loop=nb_shuffles),
                                 inf,
                                 dict(),
                                 ) for _ in range(required)]
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
            if self.verbose:
                self.log_info('About to send %s jobs to thread pool of size %s ...' % (number_format(sample_size),
                                                                                       number_format(pool_size)))
            chunk_size = ceil(sample_size / pool_size) if self.chunk_size <= 0 else self.chunk_size
            results = pool.map(partial(cls.__job__, self, nb_shuffles),
                               range(sample_size),
                               chunksize=chunk_size)
            if self.verbose:
                self.log_info('... received all jobs from thread pool')
            consecutive_timeout = 0
            sample = 0
            nb_not_optimal = 0
            cost_list = list()
            optimal_cost_list = list()
            expanded_nodes_list = list()
            run_time_list = list()
            for solution in results:
                if self.log_solution:
                    self.log_info(solution)
                cost = solution.cost
                path = solution.path
                expanded_nodes = solution.expanded_nodes
                run_time = solution.run_time
                timed_out = solution.time_out
                index = solution.additional_info.get('index')
                assert cost < 0 or isinf(cost) or cost <= len(path), 'WTF?'
                if len(self.shuffles_data[nb_shuffles][index]) < 3:
                    self.shuffles_data[nb_shuffles][index] = (self.shuffles_data[nb_shuffles][index][0],
                                                              self.shuffles_data[nb_shuffles][index][1],
                                                              dict())
                if timed_out:
                    consecutive_timeout += 1
                    nb_timeout += 1
                else:
                    if self.known_to_be_optimal():
                        if self.shuffles_data:
                            stored_cost = self.shuffles_data[nb_shuffles][index][1]
                            assert isinf(stored_cost) or stored_cost == cost, \
                                'Optimal solver found cost = %d vs stored cost = %d for puzzle %s' % (cost,
                                                                                                      stored_cost,
                                                                                                      solution.puzzle)
                            self.shuffles_data[nb_shuffles][index] = (self.shuffles_data[nb_shuffles][index][0],
                                                                      cost,
                                                                      self.shuffles_data[nb_shuffles][index][2])
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
                self.shuffles_data[nb_shuffles][index][2][self.get_name()] = solution
                sample += 1
                if not isinf(cost) and not isnan(cost):
                    total_cost += cost
                    max_cost = max(max_cost, cost)
                    cost_list.append(cost)
                    optimal_cost_list.append(optimal_cost)
                total_expanded_nodes += expanded_nodes
                expanded_nodes_list.append(expanded_nodes)
                max_expanded_nodes = max(max_expanded_nodes, expanded_nodes)
                total_run_time += run_time
                run_time_list.append(run_time)
                max_run_time = max(max_run_time, run_time)
                if self.max_consecutive_timeout and consecutive_timeout >= self.max_consecutive_timeout:
                    self.log_debug('break out for nb_shuffles=', nb_shuffles,
                                   'as timed-out/error-ed %d times' % self.max_consecutive_timeout)
                    early_breakout = True
                    break
            nb_solved = sample - nb_timeout
            if 0 == nb_solved:
                nb_solved = float('nan')
            avg_cost = round(total_cost / nb_solved, 1)
            median_cost = median(cost_list)
            max_cost = max(max_cost, avg_cost)
            avg_expanded_nodes = round(total_expanded_nodes / sample, 0)
            median_expanded_nodes = median(expanded_nodes_list)
            max_expanded_nodes = max(max_expanded_nodes, avg_expanded_nodes)
            avg_run_time = round(total_run_time / sample, 3)
            median_run_time = median(run_time_list)
            max_run_time = max(max_run_time, avg_run_time)
            res[cls.nb_samples] = sample
            res[cls.avg_cost] = avg_cost
            res[cls.median_cost] = median_cost
            res[cls.max_cost] = max_cost
            res[cls.avg_expanded_nodes] = avg_expanded_nodes
            res[cls.median_expanded_nodes] = median_expanded_nodes
            res[cls.max_expanded_nodes] = max_expanded_nodes
            res[cls.nb_timeout] = nb_timeout
            res[cls.avg_run_time] = nan if isnan(avg_run_time) else int(avg_run_time * 1000)
            res[cls.median_run_time] = int(median_run_time * 1000)
            res[cls.max_run_time] = nan if isnan(max_run_time) else int(max_run_time * 1000)
            res[cls.pct_solved] = int(100 * nb_solved / self.nb_samples)
            res[cls.pct_optimal] = int(100 * (sample - nb_not_optimal) / self.nb_samples)
            optimality_score = [0, 0]
            for opt_c, c in zip(optimal_cost_list, cost_list):
                if isnan(opt_c) or isinf(opt_c) or isnan(c) or isinf(c):
                    continue
                optimality_score[0] += opt_c
                optimality_score[1] += c
            res[cls.optimality_score] = int(100 * optimality_score[0] / optimality_score[1]) if optimality_score[1] != 0 else float('nan')
            performance = concat([performance,
                                  Series(res).to_frame().transpose()],
                                 ignore_index=True)
            self.log_info(performance)
        if self.shuffles_file_name:
            self.log_info('About to save shuffles data ...')
            to_pickle(self.all_shuffles_data, self.shuffles_file_name)
            self.log_info('... done saving all shuffles data to \'%s\'' % self.shuffles_file_name)
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
        performance = performance.sort_values(subset).drop_duplicates(subset=subset, keep='last')
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
        try:
            if self.do_plot == self.action_type:
                self.plot_performance()
            elif self.do_solve == self.action_type:
                puzzle = Puzzle.factory(**self.get_config())
                nb_shuffles = config.get(self.__class__.nb_shuffles, None)
                if nb_shuffles is not None:
                    puzzle = puzzle.apply_random_moves(nb_moves=nb_shuffles)
                return self.solve(puzzle)
            elif self.do_performance_test == self.action_type:
                self.performance_test()
            elif self.do_cleanup_performance_file == self.action_type:
                self.cleanup_perf_file()
            elif self.do_cleanup_shuffles_file == self.action_type:
                self.cleanup_shuffles_file()
            else:
                raise NotImplementedError('Unknown action_type [%s]' % self.action_type)
        except KeyboardInterrupt:
            self.log_warning('KeyboardInterrupt raised')

    def cleanup_perf_file(self):
        try:
            remove_file(self.performance_file_name)
            self.log_info('Removed \'%s\'' % self.performance_file_name)
        except FileNotFoundError as error:
            self.log_warning('Could not remove \'%s\':' % self.performance_file_name,
                             error)

    def cleanup_shuffles_file(self):
        try:
            remove_file(self.shuffles_file_name)
            self.log_info('Removed \'%s\'' % self.shuffles_file_name)
        except FileNotFoundError as error:
            self.log_warning('Could not remove \'%s\':' % self.shuffles_file_name,
                             error)

    def plot_performance(self):
        cls = self.__class__
        try:
            performance = read_pickle(self.performance_file_name)
        except FileNotFoundError:
            self.log_error('Cannot find \'%s\'. Did you really want to plot rather than solve?' %
                           self.performance_file_name)
            return
        from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic

        def short_name(solver_name):
            pos = solver_name.find(DeepLearningHeuristic.__name__)
            if pos < 0:
                return solver_name
            solver_name = solver_name[pos:]
            solver_name = solver_name[solver_name.find('[') + 1:solver_name.find(']')].replace('.pkl', '')
            solver_name = get_model_file_name(puzzle_type=self.get_puzzle_type(),
                                              dimension=self.get_puzzle_dimension(),
                                              model_name=solver_name)
            solver_name = DeepLearningHeuristic.short_name(solver_name)
            return solver_name
        performance.loc[:, Solver.solver_name] = performance[Solver.solver_name].apply(short_name)
        self.log_info(performance)
        assert 1 == len(set(performance.puzzle_type))
        assert 1 == len(set(performance.puzzle_dimension))
        assert inf in performance[Solver.nb_shuffles].values, \
            'Fix code so it uses normal axes if not inf in there'
        shuffle_max = performance[Solver.nb_shuffles].replace(inf, -1).max() * 2
        performance.loc[:, Solver.nb_shuffles] = \
            performance[Solver.nb_shuffles].replace(inf, shuffle_max)
        pprint(performance)
        y = [_ for _ in self.performance_metrics if _ in performance.columns]
        if self.performance_metrics != y:
            self.log_info('Will ignore following perf metrics which cannot be found in data: ',
                          [_ for _ in self.performance_metrics if _ not in performance.columns])
        puzzle_type = self.get_puzzle_type()
        puzzle_dimension = self.get_puzzle_dimension()
        title = {cls.puzzle_type: puzzle_type,
                 cls.puzzle_dimension: puzzle_dimension,
                 cls.performance_file_name: self.performance_file_name}
        fields_to_add = [cls.nb_samples, cls.time_out]
        for field in fields_to_add:
            title[field] = self.get_config()[field]
        fig = plt.figure(self.performance_file_name,
                         figsize=tuple(self.fig_size))
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plt.axis('off')
        title = pformat(title)
        plt.title(title, fontname='Consolas')
        n_cols = ceil(len(y) / 2)
        n_rows = ceil(len(y) / n_cols)
        sps = GridSpec(n_rows, n_cols, figure=fig)
        gb = performance.groupby(Solver.solver_name)
        max_shuffle = max(performance[Solver.nb_shuffles])
        markers = cycle(['x', '|', '.', '+', '1', 'v', '$f$', '$a$', '$t$'])
        markers = {sn: next(markers) for sn in set(performance[cls.solver_name])}
        labels_shown = False
        for r, c in product(range(n_rows), range(n_cols)):
            index = r * n_cols + c
            if index >= len(y):
                continue
            what = y[index]
            bax = brokenaxes(xlims=((0, max_shuffle/2 + 1),
                                    (max_shuffle - 1.5, max_shuffle + 1.5)),
                             subplot_spec=sps[r, c])
            if r == n_rows - 1:
                bax.set_xlabel(Solver.nb_shuffles)
            ticks = bax.get_xticks()
            labels = [['%d' % t for t in ticks[0]],
                      ['\u221e' for _ in ticks[1]]]
            bax.set_xticks('whatahorriblehack', 2, ticks, labels)
            for sn, grp in gb:
                assert len(grp[Solver.nb_shuffles]) == len(grp[what]), \
                    'Issue with %s len(x) = %d != len(y) = %d' % (sn,
                                                                  len(grp[Solver.nb_shuffles]),
                                                                  len(grp[what]))
                bax.scatter(x=Solver.nb_shuffles,
                            y=what,
                            data=grp,
                            label=sn,
                            marker=markers[sn],
                            linewidths=2,
                            s=50)
            (handles, labels) = bax.get_legend_handles_labels()[0]
            if what in [cls.avg_expanded_nodes,
                        cls.avg_run_time,
                        cls.median_expanded_nodes,
                        cls.median_run_time,
                        ]:
                bax.set_yscale('log')
                bax.set_ylabel(what + ' (log scale)')
            else:
                bax.set_ylabel(what)
            if what in [cls.pct_solved] and not labels_shown:
                """ Ideally we can show the labels on one of these so we don't have to
                display at top where it might overlap with the title. """
                bax.legend(loc=self.loc)
                labels_shown = True
        if not labels_shown:
            fig.legend(handles, labels, loc='upper center')
        plt.show()

########################################################################################################################
