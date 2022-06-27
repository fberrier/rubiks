########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from rubiks.search.searchstrategy import SearchStrategy, Node
########################################################################################################################


class DepthFirstSearch(SearchStrategy):
    """ Implements the Depth First Search, implemented recursively. Since by
    default Python will raise an exception when the call stack goes beyond some
    size (e.g. 1000), when the problem is large it will raise an exception if
    no shorter solution is found within the time out.
    Optionally we can also make the limit on the depth smaller than what Python
    would enforce by itself, via the limit parameter.
    """

    limit = 'limit'

    max_limit = 100

    @classmethod
    def populate_parser_impl(cls, parser):
        cls.add_argument(parser,
                         field=cls.limit,
                         type=int,
                         default=cls.max_limit)

    def __init__(self, initial_state, **kw_args):
        SearchStrategy.__init__(self, initial_state, **kw_args)
        self.explored = set()

    def get_name(self):
        name = SearchStrategy.get_name(self)
        if self.limit < self.max_limit:
            name += '[limit=%d]' % self.limit
        return name

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
            raise RecursionError('RecursionError')
        return solution
    
########################################################################################################################

