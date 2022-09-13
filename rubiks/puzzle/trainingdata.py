########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from functools import partial
from math import inf, ceil
from multiprocessing import Pool
from pandas import read_pickle, Series
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzled import Puzzled
from rubiks.solvers.solver import Solver
from rubiks.utils.utils import to_pickle, get_training_file_name, remove_file, is_inf, number_format, pformat
########################################################################################################################


class TrainingData(Loggable, Puzzled):

    training_data_file_name = 'training_data_file_name'

    solutions_tag = 'solutions_tag'
    puzzles_tag = 'puzzles_tag'
    nb_cpus = 'nb_cpus'
    time_out = 'time_out'
    chunk_size = 'chunk_size'
    verbose = 'verbose'
    save_full_puzzles = 'save_full_puzzles'

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
                         field=cls.chunk_size,
                         type=int,
                         default=1)
        cls.add_argument(parser,
                         field=cls.time_out,
                         default=inf)
        cls.add_argument(parser,
                         cls.verbose,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         cls.save_full_puzzles,
                         default=False,
                         action=cls.store_true)

    def __init__(self, **kw_args):
        Loggable.__init__(self, **kw_args)
        Puzzled.__init__(self, **kw_args)
        if not is_inf(self.time_out):
            self.time_out = int(self.time_out)
        if not self.training_data_file_name:
            self.training_data_file_name = get_training_file_name(puzzle_type=self.get_puzzle_type(),
                                                                  dimension=self.get_puzzle_dimension())
        self.data = None
        self.counts = dict()
        self.optimal_solver = None

    def get_solution(self, puzzle_index):
        if self.verbose:
            print('Solving puzzle # ', number_format(puzzle_index[1]), '...')
        solution = self.optimal_solver.solve(puzzle_index[0])
        if self.verbose:
            print('... done for puzzle # ', number_format(puzzle_index[1]))
        return solution

    def generate(self,
                 nb_shuffles,
                 nb_sequences,
                 min_no_loop=None,
                 repeat=1,
                 **kw_args):
        if nb_sequences <= 0 or repeat <= 0:
            return
        pool_size = min(self.nb_cpus, nb_sequences)
        pool = Pool(pool_size)
        interrupted = False
        chunk_size = ceil(nb_sequences / pool_size) if self.chunk_size <= 0 else self.chunk_size
        try:
            self.optimal_solver = Solver.factory(**{**self.get_config(),
                                                 **self.get_goal().optimal_solver_config(),
                                                 self.__class__.time_out: self.time_out})
            puzzles = [self.get_goal().apply_random_moves(nb_moves=nb_shuffles,
                                                          min_no_loop=min_no_loop) for _ in range(nb_sequences)]
            puzzles = zip(puzzles, range(1, nb_sequences + 1))
            if pool_size > 1:
                solutions = pool.map(partial(self.__class__.get_solution, self),
                                     puzzles,
                                     chunksize=chunk_size)
            else:
                solutions = [self.get_solution(puzzle_index) for puzzle_index in puzzles]
        except KeyboardInterrupt:
            self.log_warning('Interrupted')
            solutions = list()
            interrupted = True
        pool.close()
        pool.join()
        self.fetch_data()
        for solution in solutions:
            if not solution.success:
                continue
            optimal_cost = solution.cost
            puzzle = solution.puzzle.clone()
            if self.save_full_puzzles:
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
        self.remove_duplicates()
        to_pickle(self.data, self.training_data_file_name)
        self.print_data_stats()
        if repeat > 1 and not interrupted:
            self.generate(nb_shuffles=nb_shuffles,
                          nb_sequences=nb_sequences,
                          min_no_loop=min_no_loop,
                          repeat=repeat - 1)

    def print_data_stats(self):
        s = Series(data={optimal_cost: len(solutions) for optimal_cost, solutions in
                         self.data[self.solutions_tag].items()},
                   name='# SEQUENCES').sort_index().to_frame()
        if s.empty:
            return
        s.index.name = 'SEQUENCE MAX COST'
        s = s.reset_index(drop=False)
        self.log_info(pformat(s, showindex=False))
        self.log_info({'TOTAL SEQUENCES': sum(s['# SEQUENCES'])})

    def get(self, nb_shuffles):
        self.fetch_data()
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
        count += 1
        count %= len(data[nb_shuffles])
        self.counts[nb_shuffles] = count
        return solution

    def fetch_data(self):
        if self.data is not None:
            return
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

