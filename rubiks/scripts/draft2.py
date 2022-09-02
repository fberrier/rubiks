########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from pandas import DataFrame, Series
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.puzzle.puzzle import Puzzle
from rubiks.heuristics.heuristic import Heuristic
from rubiks.solvers.solver import Solver, Solution
from rubiks.utils.utils import ms_format, number_format
########################################################################################################################


if '__main__' == __name__:
    """ Just create a logger to print some stuff in this script """
    logger = Loggable(name=__file__)
    sp = Puzzle.factory(puzzle_type='sliding_puzzle', tiles=[[8, 6, 7], [2, 5, 4], [3, 0, 1]])
    solver_kw_args = {'puzzle_type': 'sliding_puzzle',
                      'n': 3,
                      'time_out': 7200}
    folder = 'C:/Users/franc/rubiksdata/models/sliding_puzzle/3_3/'
    solvers = [Solver.factory(solver_type=Solver.astar, heuristic_type=Heuristic.manhattan, plus=False, **solver_kw_args),
               Solver.factory(solver_type=Solver.astar, heuristic_type=Heuristic.manhattan, plus=True, **solver_kw_args),
               Solver.factory(solver_type=Solver.astar,
                              heuristic_type=Heuristic.perfect,
                              model_file_name=folder + 'perfect.pkl',
                              **solver_kw_args),
               Solver.factory(solver_type=Solver.astar,
                              heuristic_type=Heuristic.deep_learning,
                              model_file_name=folder + 'deep_learner_rms_prop_exponential_scheduler_999900000gamma_100seq_31shf_10000epc_tng_ntk_updt_15minshf_31maxshf_captgt_fully_connected_600_300_100_ohe.pkl',
                              **solver_kw_args),
               Solver.factory(solver_type=Solver.astar,
                              heuristic_type=Heuristic.deep_learning,
                              model_file_name=folder + 'deep_learner_rms_prop_exponential_scheduler_999900000gamma_100seq_31shf_10000epc_tng_ntk_updt_15minshf_31maxshf_captgt_convolutional_81_300_300_600_300_100_ohe.pkl',
                              **solver_kw_args),
               Solver.factory(solver_type=Solver.astar,
                              heuristic_type=Heuristic.deep_learning,
                              model_file_name=folder + 'deep_reinforcement_learner_rms_prop_exponential_scheduler_999900000gamma_100seq_50shf_25000epc_tng_ntk_updt_captgt_fully_connected_600_300_100_ohe.pkl',
                              **solver_kw_args),
               Solver.factory(solver_type=Solver.astar,
                              heuristic_type=Heuristic.deep_q_learning,
                              model_file_name=folder + 'deep_q_learner_rms_prop_exponential_scheduler_999900000gamma_100seq_50shf_25000epc_tng_ntk_updt_captgt_fully_connected_600_300_100_ohe.pkl',
                              **solver_kw_args),
               *[Solver.factory(solver_type=Solver.mcts,
                                c=c,
                                trim_tree=True,
                                heuristic_type=Heuristic.deep_q_learning,
                                model_file_name=folder + 'deep_q_learner_rms_prop_exponential_scheduler_999900000gamma_100seq_50shf_25000epc_tng_ntk_updt_captgt_fully_connected_600_300_100_ohe.pkl',
                                **solver_kw_args) for c in range(0, 70)],
               Solver.factory(solver_type=Solver.naive, heuristic_type=Heuristic.perfect, **solver_kw_args),
               Solver.factory(solver_type=Solver.bfs, **solver_kw_args),
               Solver.factory(solver_type=Solver.dfs, limit=32, **solver_kw_args),
               ]
    solutions = list()
    for solver in solvers:
        logger.log_info('Starting solving with ', solver.get_name(), '...')
        try:
            solution = solver.solve(sp)
            if solution.failed():
                logger.log_error(' ... failed:', solution.additional_info)
                continue
        except Exception as error:
            logger.log_error(error)
            continue
        solutions.append(solution)
        logger.log_info(' ... done in ', ms_format(solutions[-1][Solution.run_time]))
    columns = [Solution.solver_name, Solution.cost, Solution.expanded_nodes, Solution.run_time]
    data = DataFrame(columns=columns)
    for solver, solution in zip(solvers, solutions):
        s = dict()
        s[Solution.solver_name] = solver.get_name()
        s[Solution.expanded_nodes] = number_format(solution.expanded_nodes)
        s[Solution.cost] = solution.cost
        s[Solution.run_time] = ms_format(solution.run_time)
        data = data.append(Series(s), ignore_index=True)
    logger.log_info(data)

########################################################################################################################

