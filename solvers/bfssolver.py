########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.search.strategies import BreadthFirstSearch
from rubiks.solvers.solver import Solver
########################################################################################################################


class BFSSolver(Solver):

    def solve_impl(self, puzzle, time_out):
        #print('initial_state: ', puzzle)
        strat = BreadthFirstSearch(puzzle, time_out=time_out)
        strat.solve()
        #print('cost: ', strat.get_path_cost())
        solution = strat.get_path()
        puzzle = puzzle.clone()
        for move in solution:
            puzzle = puzzle.apply(move)
            #print(puzzle)
        return strat.get_path_cost(), solution

########################################################################################################################
