########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from unittest import TestCase
########################################################################################################################
from rubiks.search.searchstrategy import SearchStrategy
from rubiks.search.bfsstrategy import BreadthFirstSearch
from rubiks.search.dfsstrategy import DepthFirstSearch
from rubiks.heuristics.manhattan import Manhattan
from rubiks.utils.utils import snake_case
########################################################################################################################


class TestSearchStrategy(TestCase):

    def test_search_strategy_direct_construct(self):
        search = BreadthFirstSearch(initial_state=None)
        self.assertEqual(snake_case(BreadthFirstSearch.__name__),
                         search.search_strategy_type)

    def test_search_strategy(self):
        for search_strategy_type in SearchStrategy.known_search_strategy_types:
            search = SearchStrategy.factory(search_strategy_type=search_strategy_type,
                                            initial_state=None,
                                            heuristic_type=Manhattan(n=3, m=3))
            if search_strategy_type == SearchStrategy.dfs:
                self.assertEqual(DepthFirstSearch.max_limit, search.limit)
            self.assertEqual(search.search_strategy_type, search_strategy_type)

########################################################################################################################
