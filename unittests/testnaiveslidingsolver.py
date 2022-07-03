########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from math import inf
from numpy.random import randint
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.solvers.solver import Solver
from rubiks.solvers.naiveslidingsolver import NaiveSlidingSolver
from rubiks.puzzle.puzzle import Puzzle
from rubiks.puzzle.slidingpuzzle import SlidingPuzzle
########################################################################################################################


class TestNaiveSlidingSolver(TestCase):

    def test_reduce_4_4_to_3_3(self):
        logger = Loggable(name='test_reduce_4_4_to_3_3')
        how_many = 10000
        n = 4
        for _ in range(how_many):
            puzzle_type = Puzzle.sliding_puzzle
            solver_type = Solver.naive
            vars = locals()
            vars.pop('self', None)
            sliding_puzzle = Puzzle.factory(**vars).apply_random_moves(inf)
            solver = Solver.factory(**vars)
            logger.log_info('Reducing puzzle # ', _ + 1)
            logger.log_info(sliding_puzzle)
            solution = solver.reduce(sliding_puzzle,
                                     max_n=3,
                                     max_m=3)
            sliding_puzzle = solution.apply(sliding_puzzle)
            logger.log_info(sliding_puzzle)
            expected_column = list(range(1, sliding_puzzle.m * sliding_puzzle.n, sliding_puzzle.m))
            actual_column = sliding_puzzle.tiles.flatten()[::sliding_puzzle.m].tolist()
            logger.log_info({'expected_column': expected_column,
                             'actual_column': actual_column})
            self.assertEqual(expected_column, actual_column)
            expected_row = list(range(1, sliding_puzzle.m + 1))
            actual_row = sliding_puzzle.tiles.flatten()[:sliding_puzzle.m].tolist()
            logger.log_info({'expected_row': expected_row,
                             'actual_row': actual_row})
            self.assertEqual(expected_row, actual_row)

    def solve_left_column(self, name, how_many, **kw_args):
        logger = Loggable(name=name)
        for _ in range(how_many):
            puzzle_type = Puzzle.sliding_puzzle
            solver_type = Solver.naive
            vars = {**locals(), **kw_args}
            vars.pop('self', None)
            sliding_puzzle = Puzzle.factory(**vars).apply_random_moves(inf)
            solver = Solver.factory(**vars)
            logger.log_info('Solving left column for puzzle # ', _ + 1)
            logger.log_info(sliding_puzzle)
            moves = solver.solve_left_col(sliding_puzzle)
            #print(sliding_puzzle)
            expected = list(range(1, sliding_puzzle.m * sliding_puzzle.n, sliding_puzzle.m))
            actual = sliding_puzzle.tiles.flatten()[::sliding_puzzle.m].tolist()
            logger.log_info({'expected': expected,
                             'actual': actual})
            self.assertEqual(expected, actual)

    def test_solve_left_row_4_4(self):
        self.solve_left_column(name='test_solve_left_row_4_4',
                               how_many=1000,
                               n=4)

    def solve_top_row(self, name, how_many, **kw_args):
        logger = Loggable(name=name)
        for _ in range(how_many):
            puzzle_type = Puzzle.sliding_puzzle
            solver_type = Solver.naive
            vars = {**locals(), **kw_args}
            vars.pop('self', None)
            sliding_puzzle = Puzzle.factory(**vars).apply_random_moves(inf)
            solver = Solver.factory(**vars)
            #print(sliding_puzzle)
            moves = solver.solve_top_row(sliding_puzzle)
            #(sliding_puzzle)
            expected = list(range(1, sliding_puzzle.m + 1))
            actual = sliding_puzzle.tiles.flatten()[:sliding_puzzle.m].tolist()
            logger.log_info({'expected': expected,
                             'actual': actual})
            self.assertEqual(expected, actual)

    def test_solve_top_row_4_4(self):
        self.solve_top_row(name='test_solve_top_row_4_4',
                           how_many=1000,
                           n=4)

    def solve_all_puzzles(self, name, **kw_args):
        logger = Loggable(name=name)
        puzzle_type = Puzzle.sliding_puzzle
        solver_type = Solver.naive
        vars = locals()
        vars.update(kw_args)
        max_count = kw_args.get('max_count', inf)
        vars.pop('self', None)
        vars.pop('max_count', None)
        random_puzzles = kw_args.get('random_puzzles', False)
        vars.pop('random_puzzles', None)
        count = 0

        def get_puzzles():
            if not random_puzzles:
                for sliding_puzzle in SlidingPuzzle.generate_all_puzzles(**vars):
                    yield sliding_puzzle
            else:
                while True:
                    yield SlidingPuzzle.factory(**vars).perfect_shuffle()
        for sliding_puzzle in get_puzzles():
            count += 1
            logger.log_info('Solving puzzle # ', count)
            solver = Solver.factory(**vars)
            logger.log_info(sliding_puzzle)
            solution = solver.solve(sliding_puzzle)
            logger.log_info(solution.apply(sliding_puzzle))
            self.assertTrue(solution.apply(sliding_puzzle).is_goal())
            if count >= max_count:
                return
        self.assertEqual(count, min(max_count, sliding_puzzle.possible_puzzles_nb()))
        logger.log_info('count: ', count)

    def test_solve_2_2(self):
        self.solve_all_puzzles(name='test_solve_2_2',
                               n=2)

    def test_solve_2_3(self):
        self.solve_all_puzzles(name='test_solve_2_3',
                               n=2,
                               m=3)

    def test_solve_3_3(self):
        self.solve_all_puzzles(name='test_solve_3_3',
                               n=3,
                               max_count=5000,
                               random_puzzles=True)

    def test_solve_3_4(self):
        self.solve_all_puzzles(name='test_solve_3_4',
                               n=3,
                               m=4,
                               max_count=5000,
                               random_puzzles=True)

    def test_solve_4_4(self):
        self.solve_all_puzzles(name='test_solve_4_4',
                               n=4,
                               max_count=2500,
                               random_puzzles=True)

    def test_naive_sliding_solver_moves_to_right_of(self):
        logger = Loggable(name='test_naive_sliding_solver_moves_to_right_of')
        #logger.log_info('from_below=', from_below)
        for _ in range(10000):
            n = randint(2, 10)
            m = randint(2, 10)
            sliding_puzzle = SlidingPuzzle(n=n, m=m).apply_random_moves(inf)
            #logger.log_info(sliding_puzzle)
            empty = sliding_puzzle.empty
            tile = empty
            while tile == empty:
                tile = (randint(0, sliding_puzzle.n), randint(0, sliding_puzzle.m))
            #logger.log_info(empty)
            #logger.log_info('tile:', tile, ': ', sliding_puzzle.tiles[tile[0]][tile[1]].item())
            if tile[1] == sliding_puzzle.m - 1:
                #logger.log_info('Can\'t go right of that tile')
                continue
            moves = list()
            NaiveSlidingSolver.moves_to_right_of(sliding_puzzle, tile, moves)
            sliding_puzzle = sliding_puzzle.apply_moves(moves)
            #logger.log_info(sliding_puzzle)
            assert sliding_puzzle.empty[0] == tile[0]
            assert sliding_puzzle.empty[1] == tile[1] + 1

    def test_naive_sliding_solver_moves_to_left_of(self):
        logger = Loggable(name='test_naive_sliding_solver_moves_to_left_of')
        #logger.log_info('from_below=', from_below)
        for _ in range(10000):
            n = randint(2, 10)
            m = randint(2, 10)
            sliding_puzzle = SlidingPuzzle(n=n, m=m).apply_random_moves(inf)
            #logger.log_info(sliding_puzzle)
            empty = sliding_puzzle.empty
            tile = empty
            while tile == empty:
                tile = (randint(0, sliding_puzzle.n), randint(0, sliding_puzzle.m))
            #logger.log_info(empty)
            #logger.log_info('tile:', tile, ': ', sliding_puzzle.tiles[tile[0]][tile[1]].item())
            if tile[1] == 0:
                #logger.log_info('Can\'t go left of that tile')
                continue
            moves = list()
            NaiveSlidingSolver.moves_to_left_of(sliding_puzzle, tile, moves)
            sliding_puzzle = sliding_puzzle.apply_moves(moves)
            #logger.log_info(sliding_puzzle)
            assert sliding_puzzle.empty[0] == tile[0]
            assert sliding_puzzle.empty[1] == tile[1] - 1

    def test_naive_sliding_solver_moves_on_top_of(self):
        logger = Loggable(name='test_naive_sliding_solver_moves_on_top_of')
        #logger.log_info('from_right=', from_right)
        for _ in range(10000):
            n = randint(2, 10)
            m = randint(2, 10)
            sliding_puzzle = SlidingPuzzle(n=n, m=m).apply_random_moves(inf)
            #logger.log_info(sliding_puzzle)
            empty = sliding_puzzle.empty
            tile = empty
            while tile == empty:
                tile = (randint(0, sliding_puzzle.n), randint(0, sliding_puzzle.m))
            #logger.log_info('empty:', empty)
            #logger.log_info('tile:', tile, ': ', sliding_puzzle.tiles[tile[0]][tile[1]].item())
            if tile[0] == 0:
                #logger.log_info('Can\'t go up of that tile')
                continue
            moves = list()
            NaiveSlidingSolver.moves_on_top_of(sliding_puzzle, tile, moves)
            sliding_puzzle = sliding_puzzle.apply_moves(moves)
            #logger.log_info(sliding_puzzle)
            assert sliding_puzzle.empty[0] == tile[0] - 1
            assert sliding_puzzle.empty[1] == tile[1]

    def test_naive_sliding_solver_moves_below_of(self):
        logger = Loggable(name='test_naive_sliding_solver_moves_below_of')
        #logger.log_info('from_right=', from_right)
        for _ in range(10000):
            n = randint(2, 10)
            m = randint(2, 10)
            sliding_puzzle = SlidingPuzzle(n=n, m=m).apply_random_moves(inf)
            #logger.log_info(sliding_puzzle)
            empty = sliding_puzzle.empty
            tile = empty
            while tile == empty:
                tile = (randint(0, sliding_puzzle.n), randint(0, sliding_puzzle.m))
            #logger.log_info('empty:', empty)
            #logger.log_info('tile:', tile, ': ', sliding_puzzle.tiles[tile[0]][tile[1]].item())
            if tile[0] >= sliding_puzzle.n - 1:
                #logger.log_info('Can\'t go down of that tile')
                continue
            moves = list()
            NaiveSlidingSolver.moves_below_of(sliding_puzzle, tile, moves)
            sliding_puzzle = sliding_puzzle.apply_moves(moves)
            #logger.log_info(sliding_puzzle)
            assert sliding_puzzle.empty[0] == tile[0] + 1
            assert sliding_puzzle.empty[1] == tile[1]

########################################################################################################################
