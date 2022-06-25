########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from math import inf
from time import time as snap
########################################################################################################################
from rubiks.search.node import Node
from rubiks.core.factory import Factory
from rubiks.core.loggable import Loggable
########################################################################################################################


class SearchStrategy(Loggable, Factory, metaclass=ABCMeta):
    """ Base class to represent a search strategy. All the basic logic of collecting
    the optimal path and storing it, as well as the cost and time spent running the
    algo are abstracted here. There is also an optional time out functionality
    that can raise a TimeoutError if the solve() method takes more than that to
    run.
    """

    time_out = 'time_out'
    search_strategy_type = 'search_strategy_type'
    bfs = 'bfs'
    dfs = 'dfs'
    astar = 'a*'
    known_search_strategy_types = [bfs, dfs, astar]
    initial_node = 'initial_node'

    @classmethod
    def widget_types(cls):
        from rubiks.search.bfsstrategy import BreadthFirstSearch
        from rubiks.search.dfsstrategy import DepthFirstSearch
        from rubiks.search.astarstrategy import AStar
        return {cls.bfs: BreadthFirstSearch,
                cls.dfs: DepthFirstSearch,
                cls.astar: AStar}

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.search_strategy_type,
                         type=str,
                         choices=cls.known_search_strategy_types)
        cls.add_argument(parser,
                         field=cls.time_out,
                         type=float,
                         default=inf)

    def __init__(self, initial_state, **kw_args):
        Loggable.__init__(self, **kw_args)
        Factory.__init__(self, **kw_args)
        self.initial_node = Node(state=initial_state,
                                 parent=None,
                                 action=None,
                                 path_cost=0)
        self.run_time = 0
        self.actions = []
        self.cost = 0
        self.expanded_nodes = 1

    def get_run_time(self):
        return self.run_time

    def get_path_cost(self):
        return self.cost

    def get_path(self):
        return list(reversed(self.actions))

    def increment_node_count(self):
        self.expanded_nodes += 1

    def get_node_counts(self):
        return self.expanded_nodes
    
    def solve(self):
        """ Base logic is in there, concrete search algos need to implement
        the solve_impl method.
        """
        self.run_time = -snap()
        self.actions = []
        self.cost = 0
        node = self.solve_impl()
        if node is not None:
            self.cost = node.path_cost
            node = Node(state=None,
                        parent=node,
                        action=None,
                        path_cost=node.path_cost)
            while node.parent is not None:
                node = node.parent
                if node.action is not None:
                    self.actions.append(node.action)
        self.run_time += snap()
        self.run_time = int(self.run_time)

    def check_time_out(self):
        if self.time_out is None:
            return
        run_time = int(snap() + self.run_time)
        if run_time > self.time_out:
            error = 'Exceeded timeout[%ds]. Run time = %ds' \
                % (self.time_out, int(run_time))
            raise TimeoutError(error)

    def get_name(self):
        return self.__class__.__name__
        
    @abstractmethod
    def solve_impl(self):
        """ To be implemented in concrete strategies to solve the problem 
        passed in initial state 
        :return list of actions of (optimal) solution or None of failure
        """
        pass

    @classmethod
    def factory_key_name(cls):
        return cls.search_strategy_type

########################################################################################################################
