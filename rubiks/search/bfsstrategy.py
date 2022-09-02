########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
########################################################################################################################
from rubiks.search.searchstrategy import SearchStrategy, Node
########################################################################################################################


class BreadthFirstSearch(SearchStrategy):
    """ Implements the Breadth First Search, i.e. we simply explore the nodes
    from one level of depth first before going onto the next level
    """

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
            self.increment_node_count()
            for move in node.state.possible_moves():
                child = Node(state=node.state.apply(move),
                             parent=node,
                             action=move,
                             path_cost=move.cost() + node.path_cost)
                if child in frontier or child.state in explored:
                    continue
                if child.state.is_goal():
                    self.check_time_out()
                    return child
                frontier = [child] + frontier
    
########################################################################################################################

