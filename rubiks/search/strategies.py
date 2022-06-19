########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
########################################################################################################################
from sortedcontainers import SortedDict
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.search.search import SearchStrategy, Node
########################################################################################################################


class BreadthFirstSearch(SearchStrategy):
    """ Implements the Breadth First Search, i.e. we simply explore the nodes
    from one level of depth first before going onto the next level
    """

    def __init__(self, initial_state, **kw_args):
        SearchStrategy.__init__(self, initial_state, **kw_args)

    def solve_impl(self):
        """ Hopefully fairly uncontroversial implementation of BFS """
        node = self.initial_node
        if node.state.is_goal():
            return node
        explored = set()
        frontier = [node]
        while True:
            self.check_time_out()
            if 0 >= len(frontier):
                """ Failure to find a solution """
                return None
            node = frontier[-1]
            frontier = frontier[:-1]
            explored.add(node.state)
            for move in node.state.possible_moves():
                child = Node(state=node.state.apply(move),
                             parent=node,
                             action=move,
                             path_cost=move.cost() + node.path_cost)
                self.increment_node_count()
                if child in frontier or child.state in explored:
                    continue
                if child.state.is_goal():
                    self.check_time_out()
                    return child
                frontier = [child] + frontier
    
########################################################################################################################


class DepthFirstSearch(SearchStrategy):
    """ Implements the Depth First Search, implemented recursively. Since by
    default Python will raise an exception when the call stack goes beyond some
    size (e.g. 1000), when the problem is large it will raise an exception if
    no shorter solution is found within the time out.
    Optionally we can also make the limit on the depth smaller than what Python
    would enforce by itself, via the limit parameter.
    """

    max_limit = 1000000

    def __init__(self, initial_state, **kw_args):
        self.limit = int(kw_args.pop('limit', self.max_limit))
        SearchStrategy.__init__(self, initial_state, **kw_args)
        self.explored = set()

    def name(self):
        nem = SearchStrategy.name(self)
        if self.limit < self.max_limit:
            nem += '[limit=%d]' % self.limit
        return nem

    def depth_first(self, node, depth):
        """ Typical recursive function to implement DFS """
        self.check_time_out()
        # Obviously since moves are reversible, we need to check
        # for already explored nodes, otherwise we'll get
        # 'maximum recursion depth exceeded' very quickly :)
        if node.state in self.explored:
            return
        if depth > self.limit:
            return RecursionError
        self.explored.add(node.state)
        if node.state.is_goal():
            return node
        limit_reached = False
        for move in node.state.possible_moves():
            child = Node(state=node.state.apply(move),
                         parent=node,
                         action=move,
                         path_cost=move.cost() + node.path_cost)
            self.increment_node_count()
            result = self.depth_first(child, depth+1)
            if result is RecursionError:
                limit_reached = True
            elif result is not None:
                return result
        if limit_reached:
            self.explored.remove(node.state)
            return RecursionError
                
    def solve_impl(self):
        self.explored = set()
        solution = self.depth_first(self.initial_node, depth=0)
        if solution is RecursionError:
            raise solution
        return solution
    
########################################################################################################################


class AStar(SearchStrategy):
    """ Implementation of A* search algorithm. The heuristic to use is passed
    in the __init__ method at construction time.
    """

    def __init__(self, initial_state, heuristic_type, **kw_args):
        if isinstance(heuristic_type, type):
            assert issubclass(heuristic_type, Heuristic), \
                'SearchStrategy: heuristic should be a subclass of Heuristic'
            self.heuristic = heuristic_type(**kw_args)
        else:
            assert isinstance(heuristic_type, Heuristic)
            self.heuristic = heuristic_type
        SearchStrategy.__init__(self, initial_state, **kw_args)

    def name(self):
        return 'A*[%s]' % self.heuristic.__class__.__name__

    def solve_impl(self):
        """ Uncontroversial implementation of A* """
        node = self.initial_node
        g_plus_h = 0 + self.heuristic.cost_to_go(node.state) 
        frontier = SortedDict({g_plus_h: {node}})
        """ for efficient removing from the frontier, we use a reverse mapping """
        reverse_frontier = {node: g_plus_h}
        explored = set()
        while True:
            self.check_time_out()
            if 0 >= len(frontier):
                """ Failure to find a solution """
                return None
            (g_plus_h, nodes) = frontier.popitem(0)
            node = nodes.pop()
            if 0 < len(nodes):
                frontier[g_plus_h] = nodes
            if node.state.is_goal():
                self.check_time_out()
                return node
            explored.add(node.state)
            for move in node.state.possible_moves():
                node_2 = Node(state=node.state.apply(move),
                              parent=node,
                              action=move,
                              path_cost=move.cost() + node.path_cost)
                self.increment_node_count()
                g_plus_h_2 = node_2.path_cost + self.heuristic.cost_to_go(node_2.state)
                if node_2 in reverse_frontier:
                    existing_g_plus_h = reverse_frontier[node_2]
                    if existing_g_plus_h > g_plus_h_2:
                        reverse_frontier[node_2] = g_plus_h_2
                        existing_nodes = frontier[existing_g_plus_h]
                        # There is a set of nodes for that value, we remove the correct one
                        existing_nodes.remove(node_2)
                        if 0 == len(existing_nodes):
                            frontier.pop(existing_g_plus_h)
                        else:
                            frontier[existing_g_plus_h] = existing_nodes
                        if g_plus_h_2 in frontier:
                            frontier[g_plus_h_2].add(node_2)
                        else:
                            frontier[g_plus_h_2] = {node_2}
                elif node_2.state not in explored:
                    reverse_frontier[node_2] = g_plus_h_2
                    if g_plus_h_2 in frontier:
                        frontier[g_plus_h_2].add(node_2)
                    else:
                        frontier[g_plus_h_2] = {node_2}

########################################################################################################################
