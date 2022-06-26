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
from rubiks.utils.utils import is_inf, to_pickle, number_format, hms_format, pformat
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
    nb_cpus = 'nb_cpus'
    default_nb_cpus = 1
    max_puzzles = 'max_puzzles'
    regular_save = 'regular_save'
    after_round_save = 'after_round_save'
    default_regular_save = 1000
    cpu_multiplier = 'cpu_multiplier'
    default_cpu_multiplier = 10
    time_out = Solver.time_out
    heuristic_type = Heuristic.heuristic_type
    solver_type = Solver.solver_type
    puzzle_generation = 'puzzle_generation'
    random_puzzle_generation = 'random_puzzle_generation'
    permutation_puzzle_generation = 'permutation_puzzle_generation'

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
                                  cls.random_puzzle_generation],
                         type=str)
        cls.add_argument(parser,
                         field=cls.solver_type,
                         default=Solver.astar,
                         type=str)
        cls.add_argument(parser,
                         field=cls.time_out,
                         type=float,
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
                              cls.computing_time_tag: 0.}
            self.computing_time = 0
        self.highest_cost = 0 if not self.data_base[cls.data] else max(self.data_base[cls.data].values())
        self.snap = None

    def add_puzzle_to_data_base(self, puzzle, cost):
        cls = self.__class__
        h = hash(puzzle)
        if h in self.data_base[cls.data]:
            return
        self.puzzle_count += 1
        self.puzzle_count_since_save += 1
        if self.puzzle_count_since_save >= self.regular_save:
            self.puzzle_count_since_save = 0
            self.save(self.learning_file_name)
        self.data_base[cls.data][h] = cost
        if cost > self.highest_cost:
            self.highest_cost = cost
            self.data_base[cls.most_difficult_puzzle_tag] = puzzle

    def add_solution_to_data_base(self, solution):
        if is_inf(solution.cost):
            self.log_error('Timed out while solving ', solution.puzzle)
        self.add_puzzle_to_data_base(solution.puzzle, solution.cost)
        for move in solution.path:
            solution.puzzle = solution.puzzle.apply(move)
            solution.cost -= 1
            self.add_puzzle_to_data_base(solution.puzzle, solution.cost)

    def __job__(self, solver, config, puzzle):
        try:
            return solver.solve(puzzle, **config)
        except TimeoutError:
            return Solution(inf, [], inf, puzzle)

    def generate_puzzles(self):
        self.log_info('Puzzles generation process:', self.puzzle_generation)
        if self.puzzle_generation == self.permutation_puzzle_generation:
            for puzzle in self.get_puzzle_type_class().generate_all_puzzles(**self.get_config()):
                yield puzzle
        elif self.puzzle_generation == self.random_puzzle_generation:
            goal = self.get_goal()
            while True:
                yield goal.perfect_shuffle()
        else:
            raise NotImplementedError('Unknown %s [%s]' % (self.__class__.puzzle_generation,
                                                           self.puzzle_generation))

    def learn(self):
        cls = self.__class__
        solver = Solver.factory(**self.get_config())
        pool = Pool(self.nb_cpus)
        puzzles = []
        self.puzzle_count = 1
        config = self.get_config()
        self.snap = snap()
        try:
            for puzzle in self.generate_puzzles():
                if self.puzzle_count > self.max_puzzles or \
                        len(self.data_base[cls.data]) >= self.possible_puzzles_nb():
                    break
                h = hash(puzzle)
                if h in self.data_base[cls.data]:
                    continue
                if len(puzzles) < self.cpu_multiplier * self.nb_cpus:
                    puzzles.append(puzzle)
                else:
                    solutions = pool.map(partial(self.__class__.__job__,
                                                 self,
                                                 solver,
                                                 config),
                                         puzzles)
                    puzzles = [puzzle]
                    for solution in solutions:
                        self.add_solution_to_data_base(solution)
                    if self.after_round_save:
                        self.puzzle_count_since_save = 0
                        self.save(self.learning_file_name)
                if self.puzzle_count >= self.max_puzzles or \
                        len(self.data_base[cls.data]) >= self.possible_puzzles_nb():
                    break
            if puzzles and len(self.data_base[cls.data]) < self.possible_puzzles_nb():
                solutions = pool.map(partial(self.__class__.__job__,
                                             self,
                                             solver,
                                             config),
                                     puzzles)
                for solution in solutions:
                    self.add_solution_to_data_base(solution)
                if self.after_round_save:
                    self.puzzle_count_since_save = 0
                    self.save(self.learning_file_name)
        except KeyboardInterrupt:
            self.log_warning('Was interrupted. Exit and save')
        pool.close()
        pool.join()
        if self.learning_file_name:
            self.save(self.learning_file_name)

    def save(self, learning_file_name, **kwargs):
        if not learning_file_name:
            learning_file_name = self.learning_file_name
        if not learning_file_name:
            return
        cls = self.__class__
        self.data_base[cls.computing_time_tag] = self.computing_time + snap() - self.snap
        to_pickle(self.data_base, learning_file_name)
        n = len(self.data_base[cls.data])
        if n == 0:
            return
        info = {'saved puzzles': number_format(n),
                'learning_file_name': learning_file_name,
                'max cost': self.highest_cost,
                '# possible puzzles': number_format(self.possible_puzzles_nb()),
                'hardest puzzle so far': '%s' % self.data_base[cls.most_difficult_puzzle_tag],
                'computing time': hms_format(self.computing_time + snap() - self.snap),
                '# puzzles vs cost': pformat(self.puzzles_vs_cost(self.data_base[cls.data]).to_frame().transpose())}
        self.log_info(info)

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
        data = read_pickle(self.learning_file_name)[self.__class__.data]
        if not data:
            return
        total_puzzles = len(data)
        max_cost = max(data.values())
        title = '%s | %s\n# puzzles = %s\nmax cost = %s' % (self.get_puzzle_type(),
                                                            tuple(self.get_puzzle_dimension()),
                                                            number_format(total_puzzles),
                                                            number_format(max_cost))
        fig = plt.figure(self.learning_file_name, figsize=(15, 10))
        ax = fig.gca()
        self.puzzles_vs_cost(data).\
            plot(kind='bar',
                 label='# puzzles for each optimal cost',
                 color='navy')
        plt.legend()
        plt.xlabel('Optimal cost')
        plt.ylabel('# of puzzles')
        plt.title(title)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()

########################################################################################################################
