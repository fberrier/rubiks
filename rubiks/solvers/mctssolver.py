########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf
from numpy import sqrt
from time import time as snap
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.heuristics.heuristic import Heuristic
from rubiks.search.searchstrategy import SearchStrategy
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class MCTSNode(Loggable):

    def __init__(self, puzzle, parent, action, heuristic, c, nu, tree=None):
        Loggable.__init__(self, name=str(puzzle))
        self.puzzle = puzzle
        self.parent = parent
        self.action = action
        self.best_action = None
        self.heuristic = heuristic
        self.c = c
        self.nu = nu
        self.children = dict()
        self.n = dict()
        self.w = dict()
        self.l = dict()
        self.p = dict()
        self.expanded = False
        if tree is None:
            self.tree = dict()
        else:
            self.tree = tree
        self.tree[hash(self.puzzle)] = self

    def __repr__(self):
        return str(self.puzzle)

    def __hash__(self):
        return hash((hash(self.__class__.__name__), hash(self.puzzle)))

    def is_goal(self):
        return self.puzzle.is_goal()

    def expand(self):
        self.expanded = True
        possible_moves = self.puzzle.possible_moves()
        optimal_actions = self.heuristic.optimal_actions(puzzle=self.puzzle)
        for move_nb, move in enumerate(self.puzzle.theoretical_moves()):
            if move not in possible_moves:
                continue
            child = self.puzzle.apply(move)
            self.n[move_nb] = 0
            self.w[move_nb] = self.heuristic.cost_to_go(puzzle=child)
            self.l[move_nb] = 0
            self.p[move_nb] = optimal_actions[move]
            hash_child = hash(child)
            if hash_child in self.tree:
                self.children[move_nb] = self.tree[hash_child]
            else:
                self.children[move_nb] = MCTSNode(puzzle=child,
                                                  parent=self,
                                                  action=move_nb,
                                                  heuristic=self.heuristic,
                                                  c=self.c,
                                                  nu=self.nu,
                                                  tree=self.tree)
        scaling = sum(self.p.values())
        self.p = {action: proba/scaling for action, proba in self.p.items()}

    def backward_prop(self, action=None, w=None):
        if action is not None and w is not None:
            if w < self.w[action]:
                self.w[action] = w
            #self.n[action] += 1
            self.l[action] -= self.nu
        if self.parent is not None:
            self.parent.backward_prop(self.action, min(self.w.values()))

    def choose_next(self):
        best_value = -inf
        self.best_action = None
        num = sqrt(sum(self.n.values()))
        #debug_actions = dict()
        for action in self.p.keys():
            u = self.c * self.p[action] * num / (1 + self.n[action])
            q = self.w[action] + self.l[action]
            value = u - q
            if value > best_value:
                best_value = value
                self.best_action = action
            #debug_actions[action] = tuple('%.2g' % v for v in (u, -q, value))
        #self.log_info('debug_actions: ', debug_actions, ' -> best_action=', self.best_action)
        self.n[self.best_action] += 1
        self.l[self.best_action] += self.nu
        return self.children[self.best_action]

    def construct_solution(self) -> Solution:
        cost = 0
        node = self
        path = list()
        success = node.puzzle.is_goal()
        while node.parent is not None:
            action = node.action
            node = node.parent
            path.append(node.puzzle.theoretical_moves()[action])
            cost += 1
        path = list(reversed(path))
        return Solution(cost=cost,
                        path=path,
                        expanded_nodes=0,
                        puzzle=node.puzzle,
                        success=success)

    class TrimMove:

        def __init__(self, action_nb):
            self.action_nb = action_nb

        @staticmethod
        def cost():
            return 1

        def apply(self, node):
            return node.children[self.action_nb]

    def apply(self, move):
        return move.apply(self)

    def possible_moves(self):
        return [self.TrimMove(action_nb) for action_nb in self.children.keys()]

    def trim(self) -> Solution:
        """ We perform a BFS on the whole tree from init to solution to trim the tree a bit
        (might be cycles. might not be the quickest route) """
        # @todo ...so we can use BFSStrategy ... might need a few additional members functions in here
        search = SearchStrategy.factory(search_strategy_type=SearchStrategy.bfs,
                                        initial_state=self)
        search.solve()
        path = list()
        node = self
        for action in search.get_path():
            path.append(node.puzzle.theoretical_move(action.action_nb))
            node = node.apply(action)
        return Solution(cost=search.get_path_cost(),
                        path=path,
                        expanded_nodes=search.get_node_counts(),
                        puzzle=self.puzzle,
                        success=True)

########################################################################################################################


class MonteCarloTreeSearchSolver(Solver):
    """ We implement section 4.2 of the Agostinelli paper which we saved at
     rubiks.papers.SolvingTheRubiksCubeWithoutHumanKnowledge.pdf """

    c = 'c'
    nu = 'nu'
    trim_tree = 'trim_tree'

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.c,
                         type=float,
                         default=1.0)
        cls.add_argument(parser,
                         field=cls.nu,
                         type=float,
                         default=1.0)
        cls.add_argument(parser,
                         field=cls.trim_tree,
                         default=False,
                         action=cls.store_true)

    def __init__(self, **kw_args):
        Solver.__init__(self, **kw_args)
        self.run_time = None
        self.tree = None
        self.expanded_nodes = 0
        self.puzzle = None
        self.heuristic = Heuristic.factory(**self.get_config())
        self.debug_expanded_nodes = set()
        assert self.heuristic.heuristic_type == Heuristic.deep_q_learning, \
            'MonteCarloTreeSearchSolver only works with %s heuristic' % Heuristic.deep_q_learning

    def known_to_be_optimal(self):
        return False

    def solve_impl(self, puzzle, **kw_args):
        self.run_time = -snap()
        self.puzzle = puzzle
        while True:
            try:
                if self.run_time + snap() > self.time_out:
                    raise TimeoutError()
                leaf = self.go_to_leaf()
                if leaf.is_goal():
                    if not self.trim_tree:
                        solution = leaf.construct_solution()
                    else:
                        solution = self.tree.trim()
                    solution.expanded_nodes += self.expanded_nodes
                    solution.set_additional_info(solver_name=self.get_name())
                    return solution
            except TimeoutError:
                solution = Solution.failure(puzzle,
                                            time_out=True,
                                            failure_reson='time out')
                solution.expanded_nodes = self.expanded_nodes
                return solution

    def go_to_leaf(self) -> MCTSNode:
        #self.log_info('go_to_leaf')
        if self.tree is None:
            self.tree = MCTSNode(puzzle=self.puzzle,
                                 parent=None,
                                 action=None,
                                 heuristic=self.heuristic,
                                 c=self.c,
                                 nu=self.nu)
            if self.tree.is_goal():
                return self.tree
        node = self.tree
        while node.expanded:
            node = node.choose_next()
        leaf = node
        assert hash(leaf) not in self.debug_expanded_nodes, '%s already in debug_expanded_nodes' % str(leaf.puzzle)
        self.debug_expanded_nodes.add(hash(leaf))
        leaf.expand()
        leaf.backward_prop()
        self.expanded_nodes += 1
        #self.log_info('new leaf: ', leaf.puzzle)
        return leaf

    def get_name(self):
        return '%s[c=%.2g][nu=%.2g]%s[%s]' % (self.__class__.__name__,
                                              self.c,
                                              self.nu,
                                              '[trim]' if self.trim_tree else '',
                                              self.heuristic.get_name())

########################################################################################################################
