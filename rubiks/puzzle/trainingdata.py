########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from functools import partial
from math import inf
from multiprocessing import Pool
from pandas import read_pickle, Series
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzled import Puzzled
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import to_pickle, get_training_file_name, remove_file, is_inf
########################################################################################################################


class TrainingData(Loggable, Puzzled):

    training_data_file_name = 'training_data_file_name'

    solutions_tag = 'solutions_tag'
    puzzles_tag = 'puzzles_tag'
    nb_cpus = 'nb_cpus'
    time_out = 'time_out'

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.training_data_file_name,
                         type=str)
        cls.add_argument(parser,
                         field=cls.nb_cpus,
                         type=int,
                         default=1)
        cls.add_argument(parser,
                         field=cls.time_out,
                         default=inf)

    def __init__(self, **kw_args):
        Loggable.__init__(self, **kw_args)
        Puzzled.__init__(self, **kw_args)
        if not is_inf(self.time_out):
            self.time_out = int(self.time_out)
        if not self.training_data_file_name:
            self.training_data_file_name = get_training_file_name(puzzle_type=self.get_puzzle_type(),
                                                                  dimension=self.get_puzzle_dimension())
        try:
            self.data = read_pickle(self.training_data_file_name)
        except FileNotFoundError as error:
            self.log_warning('Could not fetch training data: ', error)
            self.data = None
        if not isinstance(self.data, dict):
            self.data = dict()
        if self.puzzles_tag not in self.data:
            self.data[self.puzzles_tag] = dict()
        if self.solutions_tag not in self.data:
            self.data[self.solutions_tag] = dict()
        self.counts = dict()

    @staticmethod
    def get_solution(goal, optimal_solver, nb_shuffles, min_no_loop, *args):
        puzzle = goal.apply_random_moves(nb_moves=nb_shuffles,
                                         min_no_loop=min_no_loop)
        return optimal_solver.solve(puzzle)

    def generate(self,
                 nb_shuffles,
                 nb_sequences,
                 min_no_loop=None,
                 repeat=1,
                 **kw_args):
        if nb_sequences <= 0:
            return
        pool_size = min(self.nb_cpus, nb_sequences)
        pool = Pool(pool_size)
        interrupted = False
        try:
            goal = self.get_goal()
            optimal_solver = Solver.factory(**{**self.get_config(),
                                               **goal.optimal_solver_config(),
                                               self.__class__.time_out: self.time_out})
            solutions = pool.map(partial(self.get_solution,
                                         goal,
                                         optimal_solver,
                                         nb_shuffles,
                                         min_no_loop),
                                 range(nb_sequences),
                                 chunksize=int(nb_sequences/pool_size))
        except KeyboardInterrupt:
            self.log_warning('Interrupted')
            solutions = list()
            interrupted = True
        pool.close()
        pool.join()
        for solution in solutions:
            if not solution.success:
                continue
            optimal_cost = solution.cost
            puzzle = solution.puzzle.clone()
            self.data[self.puzzles_tag][hash(puzzle)] = (puzzle, optimal_cost)
            for move in solution.path:
                puzzle = puzzle.apply(move)
                optimal_cost -= move.cost()
                self.data[self.puzzles_tag][hash(puzzle)] = (puzzle, optimal_cost)
            assert optimal_cost == 0, 'WTF?'
            if solution.cost not in self.data[self.solutions_tag]:
                self.data[self.solutions_tag][solution.cost] = [solution]
            else:
                self.data[self.solutions_tag][solution.cost].append(solution)
            self.log_debug('Added following optimal solution to training data', solution)
        self.remove_duplicates()
        to_pickle(self.data, self.training_data_file_name)
        self.log_info(Series({optimal_cost: len(solutions) for optimal_cost, solutions in
                              self.data[self.solutions_tag].items()}).sort_index())
        if repeat > 1 and not interrupted:
            self.generate(nb_shuffles=nb_shuffles,
                          nb_sequences=nb_sequences,
                          min_no_loop=min_no_loop,
                          repeat=repeat - 1)

    def get(self, nb_shuffles):
        data = self.data[self.solutions_tag]
        if nb_shuffles not in data.keys():
            min_k = min(data.keys())
            max_k = max(data.keys())
            if nb_shuffles > max_k:
                nb_shuffles = max_k
            elif nb_shuffles < min_k:
                nb_shuffles = min_k
            else:
                nb_shuffles = min(k for k in data.keys() if k < nb_shuffles)
        if nb_shuffles not in self.counts:
            self.counts[nb_shuffles] = 0
        count = self.counts[nb_shuffles]
        solution = data[nb_shuffles][count]
        self.log_debug('Returning training sequence ',
                       count,
                       ', nb_shuffles=',
                       nb_shuffles,
                       ', solution.cost=', solution.cost)
        count += 1
        count %= len(data[nb_shuffles])
        self.counts[nb_shuffles] = count
        return solution

    def remove_duplicates(self):
        self.log_debug('Removing duplicates')
        clean_solutions = dict()
        for cost, solutions in self.data[self.solutions_tag].items():
            clean_solutions[cost] = list()
            puzzles_set = set()
            for solution in solutions:
                puzzle_hash = hash(solution.puzzle)
                if puzzle_hash not in puzzles_set:
                    clean_solutions[cost].append(solution)
                puzzles_set.add(puzzle_hash)
        self.data[self.solutions_tag] = clean_solutions

    def cleanup(self):
        remove_file(self.training_data_file_name)
        self.log_info('Removed ', self.training_data_file_name)

########################################################################################################################

