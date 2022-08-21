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

    def __init__(self, puzzle, parent, action, heuristic, value, c, back_leaf_value, tree=None):
        Loggable.__init__(self, name=str(puzzle) + '%s' % hash(puzzle))
        self.puzzle = puzzle
        self.parent = parent
        """ action from parent that lead to this node """
        self.action = action
        """ action to best child """
        self.best_action = None
        self.heuristic = heuristic
        self.value = value
        self.c = c
        self.back_leaf_value = back_leaf_value
        self.children = dict()
        """ # times action has been taken """
        self.n = dict()
        """ # best value of taking this action """
        self.w = dict()
        """ # proba of action per DQL heuristic """
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
            hash_child = hash(child)
            if hash_child in self.tree:
                continue
            self.n[move_nb] = 0
            self.w[move_nb] = self.heuristic.cost_to_go(puzzle=child)
            self.p[move_nb] = optimal_actions[move]
            if hash_child in self.tree:
                self.children[move_nb] = self.tree[hash_child]
            else:
                self.children[move_nb] = MCTSNode(puzzle=child,
                                                  parent=self,
                                                  action=move_nb,
                                                  heuristic=self.heuristic,
                                                  value=self.w[move_nb],
                                                  c=self.c,
                                                  back_leaf_value=self.back_leaf_value,
                                                  tree=self.tree)
        if not self.p:
            return
        scaling = sum(self.p.values())
        self.p = {action: proba/scaling for action, proba in self.p.items()}

    def backward_prop(self, action=None, w=None):
        if action is not None:
            if w is not None:
                w += 1
                if w < self.w[action] or w == inf:
                    self.w[action] = w
                    self.value = min(self.value, min(self.w.values()))
        if self.parent is not None:
            w = inf if not self.w else self.value if self.back_leaf_value else (1 + min(self.w.values()))
            self.parent.backward_prop(self.action, w)

    def choose_next(self, depth):
        best_value = -inf
        self.best_action = None
        num = sqrt(sum(self.n.values()))
        for action in self.p.keys():
            u = self.c * self.p[action] * num / (1 + self.n[action])
            q = self.w[action]
            value = u - q
            if value > best_value:
                best_value = value
                self.best_action = action
        if self.best_action is None:
            return
        self.n[self.best_action] += 1
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
    trim_tree = 'trim_tree'
    back_leaf_value = 'back_leaf_value'

    @classmethod
    def populate_parser(cls, parser):
        cls.add_argument(parser,
                         field=cls.c,
                         type=float,
                         default=1.0)
        cls.add_argument(parser,
                         field=cls.trim_tree,
                         default=False,
                         action=cls.store_true)
        cls.add_argument(parser,
                         field=cls.back_leaf_value,
                         default=False,
                         action=cls.store_true)

    def __init__(self, **kw_args):
        Solver.__init__(self, **kw_args)
        self.run_time = None
        self.tree = None
        self.expanded_nodes = 0
        self.puzzle = None
        self.heuristic = Heuristic.factory(**self.get_config())
        assert self.heuristic.heuristic_type == Heuristic.deep_q_learning, \
            'MonteCarloTreeSearchSolver only works with %s heuristic' % Heuristic.deep_q_learning

    def re_init(self):
        self.tree = None
        self.expanded_nodes = 0
        self.puzzle = None
        self.heuristic = Heuristic.factory(**self.get_config())

    def known_to_be_optimal(self):
        return False

    def solve_impl(self, puzzle, **kw_args):
        self.run_time = -snap()
        self.re_init()
        self.puzzle = puzzle
        while True:
            try:
                if self.run_time + snap() > self.time_out:
                    raise TimeoutError()
                leaf = self.construct_new_leaf()
                if leaf.is_goal():
                    if not self.trim_tree:
                        solution = leaf.construct_solution()
                    else:
                        solution = self.tree.trim()
                    solution.expanded_nodes += self.expanded_nodes
                    solution.set_run_time(self.run_time + snap())
                    solution.set_additional_info(solver_name=self.get_name())
                    return solution
            except TimeoutError:
                solution = Solution.failure(puzzle,
                                            time_out=True,
                                            failure_reson='time out')
                solution.expanded_nodes = self.expanded_nodes
                return solution

    def construct_new_leaf(self) -> MCTSNode:
        if self.tree is None:
            self.tree = MCTSNode(puzzle=self.puzzle,
                                 parent=None,
                                 action=None,
                                 heuristic=self.heuristic,
                                 value=self.heuristic.cost_to_go(puzzle=self.puzzle),
                                 c=self.c,
                                 back_leaf_value=self.back_leaf_value)
            if self.tree.is_goal():
                return self.tree
        node = self.tree
        depth = 1
        while node.expanded:
            node = node.choose_next(depth)
            depth += 1
        leaf = node
        leaf.expand()
        leaf.backward_prop()
        self.expanded_nodes += 1
        return leaf

    def get_name(self):
        return '%s[c=%.2g]%s[%s]' % (self.__class__.__name__,
                                     self.c,
                                     '[trim]' if self.trim_tree else '',
                                     self.heuristic.get_name())

########################################################################################################################
