########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pandas import read_pickle, Series
from math import inf
from multiprocessing import Pool
from time import time as snap
########################################################################################################################
from rubiks.learners.learner import Learner
from rubiks.heuristics.heuristic import Heuristic
from rubiks.solvers.solver import Solver, Solution
from rubiks.utils.utils import is_inf, to_pickle, number_format, hms_format, pformat, s_format
from rubiks.puzzle.puzzle import Puzzle
########################################################################################################################


class PerfectLearner(Learner):
    """ Learner that goes through all the possible puzzles and solves them perfectly via an admissible heuristic
    and A* and saves all the (hash(puzzle), cost(puzzle)) for puzzle in solution.path
    (so as not to resolve already solved puzzles).
    Can obviously only be done for small dimensions
    """

    puzzle_type = Puzzle.puzzle_type
    dimension = 'dimension'
    data = 'data'
    timed_out_tag = 'timed_out'
    rerun_timed_out = 'rerun_timed_out'
    rerun_timed_out_only = 'rerun_timed_out_only'
    save_timed_out = 'save_timed_out'
    save_timed_out_max_puzzles = 'save_timed_out_max_puzzles'
    abort_after_that_many_consecutive_timed_out = 'abort_after_that_many_consecutive_timed_out'
    nb_cpus = 'nb_cpus'
    default_nb_cpus = 1
    max_puzzles = 'max_puzzles'
    regular_save = 'regular_save'
    after_round_save = 'after_round_save'
    default_regular_save = -1
    cpu_multiplier = 'cpu_multiplier'
    default_cpu_multiplier = 10
    time_out = Solver.time_out
    heuristic_type = Heuristic.heuristic_type
    solver_type = Solver.solver_type
    puzzle_generation = 'puzzle_generation'
    perfect_random_puzzle_generation = 'perfect_random_puzzle_generation'
    random_from_goal_puzzle_generation = 'random_from_goal_puzzle_generation'
    permutation_puzzle_generation = 'permutation_puzzle_generation'
    flush_timed_out_puzzles = 'flush_timed_out_puzzles'
    nb_shuffles_from_goal = 'nb_shuffles_from_goal'  # <- for random_from_goal_puzzle_generation

    most_difficult_puzzle_tag = 'most_difficult_puzzle'
    computing_time_tag = 'computing_time'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.nb_cpus,
                         type=int,
                         default=cls.default_nb_cpus)
        cls.add_argument(parser,
                         field=cls.max_puzzles,
                         type=float,
                         default=inf)
        cls.add_argument(parser,
                         field=cls.regular_save,
                         type=int,
                         default=cls.default_regular_save)
        cls.add_argument(parser,
                         field=cls.after_round_save,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.flush_timed_out_puzzles,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.save_timed_out,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.rerun_timed_out,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.rerun_timed_out_only,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.save_timed_out_max_puzzles,
                         type=int,
                         default=inf)
        cls.add_argument(parser,
                         field=cls.abort_after_that_many_consecutive_timed_out,
                         type=int,
                         default=inf)
        cls.add_argument(parser,
                         field=cls.cpu_multiplier,
                         type=int,
                         default=cls.default_cpu_multiplier)
        cls.add_argument(parser,
                         field=cls.heuristic_type,
                         default=None,
                         type=str)
        cls.add_argument(parser,
                         field=cls.puzzle_generation,
                         default=cls.permutation_puzzle_generation,
                         choices=[cls.permutation_puzzle_generation,
                                  cls.perfect_random_puzzle_generation],
                         type=str)
        cls.add_argument(parser,
                         field=cls.solver_type,
                         default=Solver.astar,
                         type=str)
        cls.add_argument(parser,
                         field=cls.time_out,
                         type=float,
                         default=0)
        cls.add_argument(parser,
                         field=cls.nb_shuffles_from_goal,
                         type=int,
                         default=0)

    def __init__(self, **kw_args):
        Learner.__init__(self, **kw_args)
        cls = self.__class__
        try:
            self.puzzle_count_since_save = 0
            if not is_inf(self.max_puzzles):
                self.max_puzzles = int(self.max_puzzles)
            else:
                self.max_puzzles = inf
            self.puzzle_count = 0
            self.data_base = read_pickle(self.learning_file_name)
            self.computing_time = self.data_base[cls.computing_time_tag]
            puzzle_type = self.data_base[PerfectLearner.puzzle_type]
            dimension = self.data_base[PerfectLearner.dimension]
            assert self.get_puzzle_type() == puzzle_type
            assert dimension == self.get_puzzle_dimension()
        except (FileNotFoundError, ModuleNotFoundError, KeyError):
            self.log_warning('Could not find (or read) data base \'%s\'' % self.learning_file_name)
            self.data_base = {cls.puzzle_type: self.get_puzzle_type(),
                              cls.dimension: self.get_puzzle_dimension(),
                              cls.most_difficult_puzzle_tag: None,
                              cls.data: dict(),
                              cls.computing_time_tag: 0.,
                              cls.timed_out_tag: dict()}
            self.computing_time = 0
        if self.flush_timed_out_puzzles:
            has_inf = any(is_inf(cost) for cost in self.data_base[cls.data].values())
            self.data_base[cls.data] = {h: cost for h, cost in self.data_base[cls.data].items() if not is_inf(cost)}
            if has_inf:
                self.data_base[cls.most_difficult_puzzle_tag] = 'Lost due to timed out flushing'
        self.highest_cost = 0 if not self.data_base[cls.data] else max(self.data_base[cls.data].values())
        self.snap = snap()
        if not self.action_type == cls.do_cleanup_learning_file:
            self.log_data()
        self.consecutive_time_outs = 0

    def add_puzzle_to_data_base(self, puzzle, cost):
        cls = self.__class__
        h = hash(puzzle)
        if h in self.data_base[cls.data]:
            return
        self.puzzle_count += 1
        self.puzzle_count_since_save += 1
        if 0 < self.regular_save <= self.puzzle_count_since_save:
            self.puzzle_count_since_save = 0
            self.save()
        self.data_base[cls.data][h] = cost
        if cost >= self.highest_cost:
            self.highest_cost = cost
            self.data_base[cls.most_difficult_puzzle_tag] = puzzle

    def add_solution_to_data_base(self, solution):
        cls = self.__class__
        if is_inf(solution.cost):
            self.consecutive_time_outs += 1
            self.log_error('Timed out while solving ', solution.puzzle)
            if self.save_timed_out and len(self.data_base[cls.timed_out_tag]) < self.save_timed_out_max_puzzles:
                self.data_base[cls.timed_out_tag][hash(solution.puzzle)] = solution.puzzle
                self.log_info('Saving puzzle to data base for later resolve')
            return
        self.consecutive_time_outs = 0
        self.add_puzzle_to_data_base(solution.puzzle, solution.cost)
        for move in solution.path:
            solution.puzzle = solution.puzzle.apply(move)
            solution.cost -= 1
            self.add_puzzle_to_data_base(solution.puzzle, solution.cost)

    def __job__(self, solver, config, puzzle):
        start = snap()
        try:
            (puzzle, index) = puzzle
            if self.verbose:
                print('Starting solving ', puzzle, ' # ', index, ' ... ')
            solution = solver.solve(puzzle, **config)
            run_time = snap() - start
            if self.verbose:
                print(' ... solved ',
                      puzzle,
                      ' # ',
                      index,
                      ' with cost ',
                      solution.cost,
                      ' in ',
                      s_format(run_time))
            return solution
        except (TimeoutError, KeyboardInterrupt):
            if self.verbose:
                if solution.failed():
                    print(' ... failed to solve ', puzzle, ' # ', index)
            return Solution(inf, list(), inf, puzzle)

    def generate_puzzles(self):
        self.log_info('Puzzles generation process:', self.puzzle_generation)
        if self.rerun_timed_out or self.rerun_timed_out_only:
            self.log_info('Will be re-running ',
                          len(self.data_base[self.__class__.timed_out_tag]),
                          ' timed-out puzzles')
            for puzzle in self.data_base[self.__class__.timed_out_tag].values():
                self.log_debug('Retrying to solve ', puzzle)
                yield puzzle
            if self.rerun_timed_out_only:
                return
        if self.puzzle_generation == self.random_from_goal_puzzle_generation:
            while True:
                puzzle = self.get_goal()
                for p in range(self.nb_shuffles_from_goal):
                    next_puzzle = puzzle.apply_random_move()
                    yield puzzle
                    puzzle = next_puzzle
        if self.puzzle_generation == self.permutation_puzzle_generation:
            for puzzle in self.get_puzzle_type_class().generate_all_puzzles(**self.get_config()):
                yield puzzle
        elif self.puzzle_generation == self.perfect_random_puzzle_generation:
            goal = self.get_goal()
            while True:
                yield goal.perfect_shuffle()
        else:
            raise NotImplementedError('Unknown %s [%s]' % (self.__class__.puzzle_generation,
                                                           self.puzzle_generation))

    def learn(self):
        cls = self.__class__
        solver = Solver.factory(**self.get_config())
        assert solver.known_to_be_optimal(), \
            'This cannot be a PerfectLearner if the solver is not optimal!'\
            ' Did you mean to use \'%s\' as heuristic_type?' % self.heuristic_type
        pool = Pool(self.nb_cpus)
        puzzles = list()
        self.puzzle_count = 1
        config = self.get_config()
        self.snap = snap()
        try:
            for puzzle in self.generate_puzzles():
                if self.consecutive_time_outs > self.abort_after_that_many_consecutive_timed_out:
                    break
                if self.puzzle_count > self.max_puzzles or \
                        len(self.data_base[cls.data]) >= self.possible_puzzles_nb():
                    break
                h = hash(puzzle)
                if h in self.data_base[cls.data]:
                    continue
                if len(puzzles) < self.cpu_multiplier * self.nb_cpus:
                    puzzles.append(puzzle)
                else:
                    n_format = number_format(len(puzzles))
                    indices = ['%s / %s' % (number_format(p + 1), n_format) for p in range(len(puzzles))]
                    puzzles = [(p, i) for p, i in zip(puzzles, indices)]
                    solutions = pool.map(partial(self.__class__.__job__,
                                                 self,
                                                 solver,
                                                 config),
                                         puzzles)
                    puzzles = [puzzle] # <- leave for next batch
                    for solution in solutions:
                        self.add_solution_to_data_base(solution)
                    if self.after_round_save:
                        self.puzzle_count_since_save = 0
                        self.save()
                if self.puzzle_count >= self.max_puzzles or \
                        len(self.data_base[cls.data]) >= self.possible_puzzles_nb():
                    break
            if puzzles and len(self.data_base[cls.data]) < self.possible_puzzles_nb() and \
                    self.consecutive_time_outs <= self.abort_after_that_many_consecutive_timed_out:
                n_format = number_format(len(puzzles))
                indices = ['%s / %s' % (number_format(p + 1),
                                        n_format) for p in range(len(puzzles))]
                puzzles = [(p, i) for p, i in zip(puzzles, indices)]
                solutions = pool.map(partial(self.__class__.__job__,
                                             self,
                                             solver,
                                             config),
                                     puzzles)
                for solution in solutions:
                    self.add_solution_to_data_base(solution)
                if self.after_round_save:
                    self.puzzle_count_since_save = 0
                    self.save()
        except KeyboardInterrupt:
            self.log_warning('Was interrupted. Exit and save')
        pool.close()
        pool.join()
        for h in self.data_base[cls.data].keys():
            # We do that at the end, don't want to mess up with it while iterating through it potentially
            self.data_base[cls.timed_out_tag].pop(h, None)
        self.save()
        if self.consecutive_time_outs > self.abort_after_that_many_consecutive_timed_out:
            self.log_warning('Aborted as too many consecutive [%d] time outs' %
                             self.abort_after_that_many_consecutive_timed_out)

    def log_data(self):
        cls = self.__class__
        data = self.data_base[cls.data]
        n = len(data)
        if n == 0:
            return
        info = {'saved puzzles': number_format(n),
                'learning_file_name': self.learning_file_name,
                'max cost': self.highest_cost,
                '# possible puzzles': number_format(self.possible_puzzles_nb()),
                'hardest puzzle so far': '%s' % self.data_base[self.most_difficult_puzzle_tag],
                'computing time': hms_format(self.computing_time + snap() - self.snap),
                '# puzzles vs cost': pformat(self.puzzles_vs_cost(data).to_frame().transpose()),
                '# timed out saved': number_format(len(self.data_base[cls.timed_out_tag])),
                }
        self.log_info(info)

    def save(self, **kwargs):
        if not self.learning_file_name:
            return
        self.data_base[self.computing_time_tag] = self.computing_time + snap() - self.snap
        to_pickle(self.data_base, self.learning_file_name)
        self.log_data()

    @staticmethod
    def puzzles_vs_cost(data):
        puzzles_per_cost = dict()
        for puzzle_hash, cost in data.items():
            if cost not in puzzles_per_cost:
                puzzles_per_cost[cost] = 0
            puzzles_per_cost[cost] += 1
        puzzles_per_cost = Series(data=puzzles_per_cost,
                                  dtype=int,
                                  name='# puzzles').sort_index()
        return puzzles_per_cost

    def plot_learning(self):
        data = read_pickle(self.learning_file_name)
        if not data[self.__class__.data]:
            return
        hardest_puzzle = str(data[self.most_difficult_puzzle_tag])
        data = data[self.__class__.data]
        solved_puzzles = len(data)
        total_puzzles = self.get_goal().possible_puzzles_nb()
        max_cost = max(data.values())
        title = {Puzzle.puzzle_type: self.get_puzzle_type(),
                 'dimension': self.get_puzzle_dimension(),
                 '# possible puzzles': number_format(total_puzzles),
                 '# solved puzzles': number_format(solved_puzzles),
                 'max cost': number_format(max_cost),
                 'hardest puzzle': hardest_puzzle,
                 }
        title = pformat(title)
        fig = plt.figure(self.learning_file_name, figsize=(15, 10))
        ax = fig.gca()
        data = self.puzzles_vs_cost(data)
        data.plot(kind='bar',
                  label='# puzzles for each optimal cost',
                  color='royalblue')
        for index, value in zip(data.index, data.values):
            plt.text(index, value, str(value))
        plt.legend()
        plt.xlabel('Optimal cost')
        plt.ylabel('# of puzzles')
        plt.title(title, fontname='Consolas')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.show()

########################################################################################################################
