########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.heuristics.manhattan import Manhattan
########################################################################################################################


class TestSliding(TestCase):

    def test_manhattan_distance_3_3(self):
        manhattan = Manhattan(n=3, m=3)
        goal = manhattan.get_goal()
        for _ in range(100):
            for nb_moves in range(1, 50):
                puzzle = goal.apply_random_moves(nb_moves, min_no_loop=nb_moves)
                heuristic_cost = manhattan.cost_to_go_from_puzzle(puzzle)
                # given the heuristic is admissible, and optimal cost is smaller than nb_moves
                # we should have heuristic_cost <= optimal_cost <= nb_moves
                self.assertGreaterEqual(nb_moves, heuristic_cost)

    def test_manhattan_distance_10_10(self):
        manhattan = Manhattan(n=10, m=10)
        goal = manhattan.get_goal()
        for _ in range(100):
            for nb_moves in range(1, 100):
                puzzle = goal.apply_random_moves(nb_moves, min_no_loop=nb_moves)
                heuristic_cost = manhattan.cost_to_go_from_puzzle(puzzle)
                # given the heuristic is admissible, and optimal cost is smaller than nb_moves
                # we should have heuristic_cost <= optimal_cost <= nb_moves
                self.assertGreaterEqual(nb_moves, heuristic_cost)

########################################################################################################################
