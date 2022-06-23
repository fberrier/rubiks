########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from functools import partial
from pandas import read_pickle
from math import inf
from multiprocessing import Pool
########################################################################################################################
from rubiks.learners.learner import Learner
from rubiks.solvers.solver import Solver, Solution
from rubiks.utils.utils import is_inf, to_pickle
########################################################################################################################


class PerfectLearner(Learner):
    """ Learner that goes through all the possible puzzles and solves them perfectly via an admissible heuristic
    and A* and saves all the (hash(puzzle), cost(puzzle)) for puzzle in solution.path
    (so as not to resolve already solved puzzles).
    Can obviously only be done for small dimensions
    """

    puzzle_type_tag = 'puzzle_type'
    dimension_tag = 'dimension'
    data_tag = 'data'

    def __init__(self, puzzle_type, data_base_file_name, **kw_args):
        Learner.__init__(self, puzzle_type, **kw_args)
        self.log_info(self.kw_args)
        try:
            self.max_puzzles = self.kw_args.get('max_puzzles', inf)
            self.regular_save = int(self.kw_args.get('regular_save', 10000))
            self.puzzle_count_since_save = 0
            self.cpu_multiplier = int(self.kw_args.get('cpu_multiplier', 10))
            if not is_inf(self.max_puzzles):
                self.max_puzzles = int(self.max_puzzles)
            else:
                self.max_puzzles = inf
            self.puzzle_count = 0
            self.data_base_file_name = data_base_file_name
            self.data_base = read_pickle(data_base_file_name)
            puzzle_type = self.data_base[PerfectLearner.puzzle_type_tag]
            dimension = self.data_base[PerfectLearner.dimension_tag]
            assert self.get_puzzle_type() == puzzle_type
            assert dimension == self.puzzle_dimension()
        except FileNotFoundError:
            self.log_warning('Could not find data base \'%s\'' % data_base_file_name)
            self.data_base = {self.puzzle_type_tag: self.get_puzzle_type(),
                              self.dimension_tag: self.puzzle_dimension(),
                              self.data_tag: dict()}

    def add_puzzle_to_data_base(self, puzzle, cost):
        h = hash(puzzle)
        if h in self.data_base[self.data_tag]:
            return
        self.puzzle_count += 1
        self.puzzle_count_since_save += 1
        if self.puzzle_count_since_save >= self.regular_save:
            self.puzzle_count_since_save = 0
            self.save(self.data_base_file_name)
        self.data_base[self.data_tag][h] = cost
        self.log_debug('Added ',
                       puzzle,
                       ' with cost ',
                       cost,
                       ' to data base. puzzle_count = ',
                       self.puzzle_count,
                       ' total data base count = ',
                       len(self.data_base[self.data_tag]),
                       ' out of ', self.possible_puzzles_nb(), ' possible puzzles.')

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
            time_out = self.kw_args.get('time_out', inf)
            if is_inf(time_out):
                time_out = inf
            else:
                time_out = int(time_out)
            return solver.solve(puzzle, time_out=time_out)
        except TimeoutError:
            return Solution(inf, [], inf, puzzle)

    def learn(self):
        solver = Solver.factory(solver_type='a*',
                                heuristic_type='manhattan',
                                **self.kw_args)
        nb_cpus = self.kw_args.get('nb_cpus', 1)
        pool = Pool(nb_cpus)
        puzzles = []
        self.puzzle_count = 0
        for puzzle in self.get_puzzle_type().generate_all_puzzles(**self.kw_args):
            h = hash(puzzle)
            if h in self.data_base[self.data_tag]:
                continue
            if len(puzzles) < self.cpu_multiplier * nb_cpus:
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
        to_pickle(self.data_base, model_file_name)
        self.log_info('Saved ',
                      len(self.data_base[self.data_tag]),
                      ' puzzles to \'%s\'' % self.data_base_file_name,
                      '. Max cost = ', max(self.data_base[self.data_tag].values()))

    @staticmethod
    def plot_learning(learning_file_name,
                      network_name=None,
                      puzzle_type=None,
                      puzzle_dimension=None):
        """ Plot something meaningful. Learning type dependent so will be implemented in derived classes """
        return

########################################################################################################################
