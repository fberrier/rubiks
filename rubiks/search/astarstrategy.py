########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from sortedcontainers import SortedDict
########################################################################################################################
from rubiks.heuristics.heuristic import Heuristic
from rubiks.search.searchstrategy import SearchStrategy, Node
########################################################################################################################


class AStar(SearchStrategy):
    """ Implementation of A* search algorithm. The heuristic to use is passed
    in the __init__ method at construction time.
    """

    heuristic_type = 'heuristic_type'

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.heuristic_type,
                         choices=Heuristic.known_heuristic_types,
                         default=Heuristic.manhattan)

    def __init__(self, puzzle, **kw_args):
        SearchStrategy.__init__(self, puzzle, **kw_args)
        self.heuristic = Heuristic.factory(**kw_args)

    def get_name(self):
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
