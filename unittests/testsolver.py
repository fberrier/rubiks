########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
from math import inf
from numpy import mean
from os.path import isfile
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.heuristics.heuristic import Heuristic
from rubiks.puzzle.puzzle import Puzzle
from rubiks.solvers.solver import Solver
from rubiks.deeplearning.deeplearning import DeepLearning
from rubiks.learners.perfectlearner import PerfectLearner
from rubiks.learners.deeplearner import DeepLearner
from rubiks.learners.deepqlearner import DeepQLearner
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.utils.utils import get_model_file_name, remove_file, ms_format
########################################################################################################################


class TestSolver(TestCase):

    def test_solver(self):
        # just instantiate an a* solver
        logger = Loggable(name='test_solver')
        solver = Solver.factory(solver_type=Solver.astar,
                                puzzle_type=Puzzle.sliding_puzzle,
                                n=2,
                                m=3)
        logger.log_info(solver.get_config())
        self.assertEqual(solver.puzzle_type, Puzzle.sliding_puzzle)
        self.assertEqual(solver.solver_type, Solver.astar)
        self.assertEqual(solver.get_puzzle_dimension(), (2, 3))

    def test_a_star_perfect_solver(self):
        logger = Loggable(name='test_a_star_perfect_solver')
        puzzle_type = Puzzle.sliding_puzzle
        dimension = (2, 2)
        # we learn first
        model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='test_a_star_perfect_solver')
        remove_file(model_file_name)
        learner = PerfectLearner(puzzle_type=Puzzle.sliding_puzzle,
                                 learning_file_name=model_file_name,
                                 solver_type=Solver.astar,
                                 heuristic_type=Heuristic.manhattan,
                                 time_out=2,
                                 n=dimension[0],
                                 m=dimension[1])
        logger.log_info(learner.get_config())
        learner.learn()
        # Then we use this learning to solve
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type=Heuristic.perfect,
                                model_file_name=model_file_name,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1])
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(10)
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        logger.log_info(solution)
        remove_file(model_file_name)

    def test_solver_a_star_manhattan(self):
        logger = Loggable(name='test_solver_a_star_manhattan')
        dimension = (3, 4)
        puzzle_type = Puzzle.sliding_puzzle
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type=Heuristic.manhattan,
                                plus=True,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1],
                                time_out=1,
                                )
        config = solver.get_config()
        logger.log_info(config)
        self.assertEqual(dimension, (config['n'], config['m']))
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(inf)
        self.assertEqual(dimension, puzzle.dimension())
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        logger.log_info(solution)
        self.assertTrue(solution.failed())
        self.assertTrue(str(solution).find('time out') >= 0)
        solution = solver.solve(puzzle=puzzle, time_out=300)
        logger.log_info(solution)
        self.assertFalse(solution.failed())
        logger.log_info(solution)

    def test_deep_reinforcement_learning_solver(self):
        logger = Loggable(name='test_deep_reinforcement_learning_solver')
        puzzle_type = Puzzle.sliding_puzzle
        dimension = (2, 2)
        # we learn first
        model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='test_deep_reinforcement_learning_solver')
        remove_file(model_file_name)
        learner = DeepReinforcementLearner(puzzle_type=Puzzle.sliding_puzzle,
                                           learning_file_name=model_file_name,
                                           solver_type=Solver.astar,
                                           n=dimension[0],
                                           m=dimension[1],
                                           nb_cpus=1,
                                           network_type=DeepLearning.fully_connected_net,
                                           layers_description=(16, 8),
                                           nb_epochs=10000,
                                           one_hot_encoding=True,
                                           nb_shuffles=12,
                                           max_target_not_increasing_epochs_pct=0.5,
                                           max_target_uptick=0.01,
                                           max_nb_target_network_update=1000,
                                           update_target_network_threshold=1e-5,
                                           learning_rate=1e-2,
                                           nb_sequences=1)
        logger.log_info(learner.get_config())
        learner.learn()
        # Then we use this learning to solve
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type=Heuristic.deep_learning,
                                model_file_name=model_file_name,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1])
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(inf)
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        self.assertTrue(solution.success)
        logger.log_info(solution)
        remove_file(model_file_name)

    def test_deep_q_learning_solver(self):
        logger = Loggable(name='test_deep_q_learning_solver')
        puzzle_type = Puzzle.sliding_puzzle
        dimension = (3, 3)
        # we learn first
        model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='test_deep_q_learning_solver')
        re_learn = False
        run_trim = True
        if re_learn or not isfile(model_file_name):
            remove_file(model_file_name)
            learner = DeepQLearner(puzzle_type=Puzzle.sliding_puzzle,
                                   learning_file_name=model_file_name,
                                   n=dimension[0],
                                   m=dimension[1],
                                   nb_cpus=3,
                                   network_type=DeepLearning.fully_connected_net,
                                   layers_description=(600, 300, 100),
                                   nb_epochs=25000,
                                   one_hot_encoding=True,
                                   nb_shuffles=50,
                                   max_target_not_increasing_epochs_pct=0.5,
                                   max_target_uptick=0.01,
                                   max_nb_target_network_update=100,
                                   update_target_network_threshold=1e-2,
                                   update_target_network_frequency=500,
                                   cap_target_at_network_count=True,
                                   learning_rate=1e-2,
                                   nb_sequences=100)
            logger.log_info(learner.get_config())
            learner.learn()
        c = 1
        logger.log_info('Running with c=', c, ' and run_trim=', run_trim)
        random_choice = False
        # Then we use this learning to solve
        solver = Solver.factory(solver_type=Solver.mcts,
                                heuristic_type=Heuristic.deep_q_learning,
                                model_file_name=model_file_name,
                                puzzle_type=puzzle_type,
                                time_out=inf,
                                n=dimension[0],
                                m=dimension[1],
                                trim_tree=False,
                                c=c,
                                random_choice=random_choice)
        solver2 = Solver.factory(solver_type=Solver.mcts,
                                 heuristic_type=Heuristic.deep_q_learning,
                                 model_file_name=model_file_name,
                                 puzzle_type=puzzle_type,
                                 time_out=inf,
                                 n=dimension[0],
                                 m=dimension[1],
                                 trim_tree=True,
                                 c=c,
                                 random_choice=random_choice)
        goal = Puzzle.factory(**solver.get_config())
        nb_moves = 20
        nb_puzzles = 100
        costs = list()
        run_times = list()
        nodes = list()
        puzzles = [goal.apply_random_moves(nb_moves) for _ in range(nb_puzzles)]
        for nb_puzzle, puzzle in enumerate(puzzles):
            nb_puzzle += 1
            solution_1 = solver.solve(puzzle=puzzle)
            #logger.log_debug(solution_1)
            self.assertTrue(solution_1.success, '%s' % solution_1)
            # with trim
            if run_trim:
                solution_2 = solver2.solve(puzzle=puzzle)
                logger.log_debug(solution_2)
                self.assertTrue(solution_2.success, '%s' % solution_2)
            else:
                solution_2 = solution_1
            if solution_1.cost > 10*nb_moves:
                logger.log_warning('Puzzle # %d ' % (nb_puzzle + 1), puzzle)
            if solution_2.cost != solution_1.cost:
                logger.log_debug('Puzzle # %d ' % (nb_puzzle + 1), puzzle)
                logger.log_debug('Puzzle %d improvement nb_moves %d solutions costs %d -> %d' % (nb_puzzle + 1,
                                                                                                nb_moves,
                                                                                                solution_1.cost,
                                                                                                solution_2.cost))
            else:
                run_time = ms_format(solution_1.run_time)
                logger.log_info('log_info %d nb_moves %d solutions costs %d nodes %d in %s' % (nb_puzzle + 1, nb_moves, solution_1.cost, solution_1.expanded_nodes, run_time))
            costs.append(min(solution_1.cost, solution_2.cost))
            run_times.append(min(solution_1.run_time, solution_2.run_time))
            nodes.append(min(solution_1.expanded_nodes, solution_2.expanded_nodes))
        logger.log_info('avg cost = ', mean(costs))
        logger.log_info('avg run_time = ', mean(run_times))
        logger.log_info('avg nodes = ', mean(nodes))
        if re_learn:
            remove_file(model_file_name)

    def test_deep_reinforcement_learning_solver_with_scheduler(self):
        logger = Loggable(name='test_deep_reinforcement_learning_solver_with_scheduler')
        puzzle_type = Puzzle.sliding_puzzle
        dimension = (2, 2)
        # we learn first
        model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='test_deep_reinforcement_learning_solver_with_scheduler')
        remove_file(model_file_name)
        learner = DeepReinforcementLearner(puzzle_type=Puzzle.sliding_puzzle,
                                           learning_file_name=model_file_name,
                                           solver_type=Solver.astar,
                                           n=dimension[0],
                                           m=dimension[1],
                                           nb_cpus=1,
                                           network_type=DeepLearning.fully_connected_net,
                                           layers_description=(16, 8),
                                           nb_epochs=10000,
                                           one_hot_encoding=True,
                                           nb_shuffles=12,
                                           max_target_not_increasing_epochs_pct=0.5,
                                           max_target_uptick=0.01,
                                           max_nb_target_network_update=1000,
                                           update_target_network_threshold=1e-5,
                                           learning_rate=1e-2,
                                           scheduler=DeepReinforcementLearner.exponential_scheduler,
                                           gamma_scheduler=0.9999,
                                           nb_sequences=1)
        logger.log_info(learner.get_config())
        learner.learn()
        # Then we use this learning to solve
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type=Heuristic.deep_learning,
                                model_file_name=model_file_name,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1])
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(inf)
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        self.assertTrue(solution.success)
        logger.log_info(solution)
        remove_file(model_file_name)

    def test_deep_learning_solver_with_scheduler(self):
        logger = Loggable(name='test_deep_learning_solver_with_scheduler')
        puzzle_type = Puzzle.sliding_puzzle
        dimension = (2, 2)
        # we learn first
        model_file_name = get_model_file_name(puzzle_type=puzzle_type,
                                              dimension=dimension,
                                              model_name='test_deep_learning_solver_with_scheduler')
        remove_file(model_file_name)
        learner = DeepLearner(puzzle_type=Puzzle.sliding_puzzle,
                              learning_file_name=model_file_name,
                              n=dimension[0],
                              m=dimension[1],
                              nb_cpus=1,
                              network_type=DeepLearning.fully_connected_net,
                              layers_description=(16, 8),
                              nb_epochs=1000,
                              one_hot_encoding=True,
                              nb_shuffles=12,
                              learning_rate=1e-2,
                              scheduler=DeepLearner.exponential_scheduler,
                              gamma_scheduler=0.9999,
                              nb_sequences=1)
        logger.log_info(learner.get_config())
        learner.learn()
        # Then we use this learning to solve
        solver = Solver.factory(solver_type=Solver.astar,
                                heuristic_type=Heuristic.deep_learning,
                                model_file_name=model_file_name,
                                puzzle_type=puzzle_type,
                                n=dimension[0],
                                m=dimension[1])
        puzzle = Puzzle.factory(**solver.get_config()).apply_random_moves(inf)
        logger.log_info(puzzle)
        solution = solver.solve(puzzle=puzzle)
        self.assertTrue(solution.success)
        logger.log_info(solution)
        remove_file(model_file_name)

    def test_kociemba_2(self):
        logger = Loggable(name='test_kociemba_2')
        for puzzle_type in [Puzzle.rubiks_cube,
                            Puzzle.watkins_cube]:
            cube = Puzzle.factory(n=2,
                                  puzzle_type=puzzle_type).apply_random_moves(100)
            solver = Solver.factory(solver_type=Solver.kociemba,
                                    puzzle_type=puzzle_type,
                                    n=cube.n)
            solution = solver.solve(cube)
            logger.log_info(solution)
            self.assertTrue(solution.success)

    def test_kociemba_3(self):
        logger = Loggable(name='test_kociemba_3')
        for puzzle_type in [Puzzle.rubiks_cube,
                            Puzzle.watkins_cube]:
            cube = Puzzle.factory(n=3,
                                  puzzle_type=puzzle_type).apply_random_moves(100)
            solver = Solver.factory(solver_type=Solver.kociemba,
                                    puzzle_type=puzzle_type,
                                    n=cube.n)
            solution = solver.solve(cube)
            logger.log_info(solution)
            self.assertTrue(solution.success)

########################################################################################################################
