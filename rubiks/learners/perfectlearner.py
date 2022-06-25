########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from functools import partial
from pandas import read_pickle
from math import inf
from multiprocessing import Pool
########################################################################################################################
from rubiks.learners.learner import Learner
from rubiks.heuristics.heuristic import Heuristic
from rubiks.solvers.solver import Solver, Solution
from rubiks.utils.utils import is_inf, to_pickle, g_not_a_pkl_file
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
    default_regular_save = 1000
    cpu_multiplier = 'cpu_multiplier'
    default_cpu_multiplier = 10
    time_out = Solver.time_out
    data_base_file_name = 'data_base_file_name'
    heuristic_type = Heuristic.heuristic_type
    solver_type = Solver.solver_type

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.nb_cpus,
                         type=int,
                         default=cls.default_nb_cpus)
        cls.add_argument(parser,
                         field=cls.max_puzzles,
                         default=inf)
        cls.add_argument(parser,
                         field=cls.regular_save,
                         default=cls.default_regular_save)
        cls.add_argument(parser,
                         field=cls.cpu_multiplier,
                         type=int,
                         default=cls.default_cpu_multiplier)
        cls.add_argument(parser,
                         field=cls.data_base_file_name,
                         type=str,
                         default=g_not_a_pkl_file)
        cls.add_argument(parser,
                         field=cls.heuristic_type,
                         default=None,
                         type=str)
        cls.add_argument(parser,
                         field=cls.solver_type,
                         default=Solver.astar,
                         type=str)
        cls.add_argument(parser,
                         field=cls.time_out,
                         default=0)

    def __init__(self, **kw_args):
        Learner.__init__(self, **kw_args)
        try:
            self.puzzle_count_since_save = 0
            if not is_inf(self.max_puzzles):
                self.max_puzzles = int(self.max_puzzles)
            else:
                self.max_puzzles = inf
            self.puzzle_count = 0
            self.data_base = read_pickle(self.data_base_file_name)
            puzzle_type = self.data_base[PerfectLearner.puzzle_type]
            dimension = self.data_base[PerfectLearner.dimension]
            assert self.get_puzzle_type() == puzzle_type
            assert dimension == self.get_puzzle_dimension()
        except FileNotFoundError:
            self.log_warning('Could not find data base \'%s\'' % self.data_base_file_name)
            cls = self.__class__
            self.data_base = {cls.puzzle_type: self.get_puzzle_type(),
                              cls.dimension: self.get_puzzle_dimension(),
                              cls.data: dict()}

    def add_puzzle_to_data_base(self, puzzle, cost):
        cls = self.__class__
        h = hash(puzzle)
        if h in self.data_base[cls.data]:
            return
        self.puzzle_count += 1
        self.puzzle_count_since_save += 1
        if self.puzzle_count_since_save >= self.regular_save:
            self.puzzle_count_since_save = 0
            self.save(self.data_base_file_name)
        self.data_base[cls.data][h] = cost
        self.log_debug('Added ',
                       puzzle,
                       ' with cost ',
                       cost,
                       ' to data base. puzzle_count = ',
                       self.puzzle_count,
                       ' total data base count = ',
                       len(self.data_base[cls.data]),
                       ' out of ',
                       self.possible_puzzles_nb(),
                       ' possible puzzles.')

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

    def learn(self):
        cls = self.__class__
        solver = Solver.factory(**self.get_config())
        pool = Pool(self.nb_cpus)
        puzzles = []
        self.puzzle_count = 1
        config = self.get_config()
        for puzzle in self.get_puzzle_type_class().generate_all_puzzles(**config):
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
            if self.puzzle_count >= self.max_puzzles:
                break
        if puzzles:
            solutions = pool.map(partial(self.__class__.__job__,
                                         self,
                                         solver,
                                         config),
                                 puzzles)
            for solution in solutions:
                self.add_solution_to_data_base(solution)
        pool.close()
        pool.join()
        self.save(self.data_base_file_name)

    def save(self, model_file_name, **kwargs):
        cls = self.__class__
        to_pickle(self.data_base, model_file_name)
        n = len(self.data_base[cls.data])
        if n <= 0:
            return
        self.log_info('Saved ',
                      n,
                      ' puzzles to \'%s\'' % self.data_base_file_name,
                      '. Max cost = ',
                      max(self.data_base[cls.data].values()))

    @staticmethod
    def plot_learning(learning_file_name,
                      network_name=None,
                      puzzle_type=None,
                      puzzle_dimension=None):
        """ Plot something meaningful. Learning type dependent so will be implemented in derived classes """
        return

########################################################################################################################
