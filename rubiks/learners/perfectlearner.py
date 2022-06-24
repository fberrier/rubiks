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
    max_puzzles = 'max_puzzles'
    regular_save = 'regular_save'
    cpu_multiplier = 'cpu_multiplier'
    time_out = Solver.time_out
    data_base_file_name = 'data_base_file_name'

    @classmethod
    def populate_parser(cls, parser):
        Learner.populate_parser(parser)  # learner_type
        Puzzle.populate_parser(parser)  # puzzle_type
        Solver.populate_parser(parser)  # time_out & max_consecutive_timeout
        cls.add_argument(parser,
                         field=cls.nb_cpus,
                         type=int,
                         default=1)
        cls.add_argument(parser,
                         field=cls.max_puzzles,
                         default=inf)
        cls.add_argument(parser,
                         field=cls.regular_save,
                         default=1000)
        cls.add_argument(parser,
                         field=cls.cpu_multiplier,
                         type=int,
                         default=10)
        cls.add_argument(parser,
                         field=cls.data_base_file_name,
                         type=str,
                         default=g_not_a_pkl_file)

    def __init__(self, puzzle_type, data_base_file_name, **kw_args):
        Learner.__init__(self, puzzle_type, **kw_args)
        self.nb_cpus = self.kw_args.get(__class__.nb_cpus, 1)
        self.log_info(self.kw_args)
        try:
            self.max_puzzles = self.kw_args.get(__class__.max_puzzles, inf)
            self.regular_save = int(self.kw_args.get(__class__.regular_save, 10000))
            self.puzzle_count_since_save = 0
            self.cpu_multiplier = int(self.kw_args.get(__class__.cpu_multiplier, 10))
            if not is_inf(self.max_puzzles):
                self.max_puzzles = int(self.max_puzzles)
            else:
                self.max_puzzles = inf
            self.puzzle_count = 0
            self.data_base_file_name = data_base_file_name
            self.data_base = read_pickle(data_base_file_name)
            puzzle_type = self.data_base[PerfectLearner.puzzle_type]
            dimension = self.data_base[PerfectLearner.dimension]
            assert self.get_puzzle_type() == puzzle_type
            assert dimension == self.get_puzzle_dimension()
        except FileNotFoundError:
            self.log_warning('Could not find data base \'%s\'' % data_base_file_name)
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

    def __job__(self, solver, puzzle):
        try:
            time_out = self.kw_args.get(__class__.time_out, inf)
            if is_inf(time_out):
                time_out = inf
            else:
                time_out = int(time_out)
            return solver.solve(puzzle, time_out=time_out)
        except TimeoutError:
            return Solution(inf, [], inf, puzzle)

    def learn(self):
        cls = self.__class__
        self.kw_args.update({Solver.solver_type: Solver.astar})
        solver = Solver.factory(heuristic_type=Heuristic.manhattan,
                                **self.kw_args)
        pool = Pool(self.nb_cpus)
        puzzles = []
        self.puzzle_count = 1
        for puzzle in self.get_puzzle_type().generate_all_puzzles(**self.kw_args):
            h = hash(puzzle)
            if h in self.data_base[cls.data]:
                continue
            if len(puzzles) < self.cpu_multiplier * self.nb_cpus:
                puzzles.append(puzzle)
            else:
                solutions = pool.map(partial(self.__class__.__job__,
                                             self,
                                             solver),
                                     puzzles)
                puzzles = [puzzle]
                for solution in solutions:
                    self.add_solution_to_data_base(solution)
            if self.puzzle_count >= self.max_puzzles:
                break
        if puzzles:
            solutions = pool.map(partial(self.__class__.__job__,
                                         self,
                                         solver),
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
