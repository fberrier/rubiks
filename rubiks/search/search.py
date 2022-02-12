########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from abc import abstractmethod, ABCMeta
from time import time as snap
########################################################################################################################
from rubiks.utils.loggable import Loggable
########################################################################################################################


class Node:
    """ Node is the data structure that we manipulate in the search algorithms.
    As long as:
    - state and parent are either None or of a type that satisfies 
      the interface of BaseState
    - action is of a type that satistifes the interface of BaseAction 
      and parent.action(action) == state
    Then we can run the search algorithms from this module on Nodes
    """

    def __init__(self,
                 state,
                 parent,
                 action,
                 path_cost: int):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __repr__(self):
        return 'State=%s, parent=%s, action=%s, path_cost=%s' % (self.state,
                                                                 self.parent,
                                                                 self.action,
                                                                 self.path_cost)

    def __hash__(self):
        """ Useful to put in containers that rely on hashes """
        return hash(self.state)

########################################################################################################################


class SearchStrategy(Loggable, metaclass=ABCMeta):
    """ Base class to represent a search strategy. All the basic logic of collecting
    the optimal path and storing it, as well as the cost and time spent running the
    algo are abstracted here. There is also an optional time out functionality
    that can raise a TimeoutError if the solve() method takes more than that to
    run.
    """

    def __init__(self, initial_state, time_out=None, **kw_args):
        self.initial_node = Node(initial_state,
                                 parent=None,
                                 action=None,
                                 path_cost=0)
        self.run_time = 0
        self.actions = []
        self.cost = 0
        self.time_out = int(time_out) if time_out is not None else time_out
        self.expanded_nodes = 1
        Loggable.__init__(self, self.name(), kw_args.pop('log_level', 'INFO'))

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
        run_time =  int(snap() + self.run_time)
        if run_time > self.time_out:
            error = 'Exceeded timeout[%ds]. Run time = %ds' \
                % (self.time_out, int(run_time))
            raise TimeoutError(error)

    def name(self):
        return self.__class__.__name__
        
    @abstractmethod
    def solve_impl(self):
        """ To be implemented in concrete strategies to solve the problem 
        passed in initial state 
        :return list of actions of (optimal) solution or None of failure
        """
        pass

########################################################################################################################
