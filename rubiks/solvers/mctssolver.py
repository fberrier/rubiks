########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf
from numpy import sqrt
from time import time as snap
########################################################################################################################
from rubiks.core.loggable import Loggable
from rubiks.heuristics.heuristic import Heuristic
from rubiks.solvers.solver import Solver, Solution
########################################################################################################################


class MCTSNode(Loggable):

    def __init__(self, puzzle, parent, action, heuristic, c, nu):
        Loggable.__init__(self, name=str(puzzle))
        self.puzzle = puzzle
        self.parent = parent
        self.action = action
        self.heuristic = heuristic
        self.c = c
        self.nu = nu
        self.children = dict()
        self.n = dict()
        self.w = dict()
        self.l = dict()
        self.p = dict()
        self.expanded = False

    def __repr__(self):
        return str(self.puzzle)

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
            self.log_info('w[', move, '] = ', self.w[move_nb])
            self.l[move_nb] = 0
            self.p[move_nb] = optimal_actions[move]
            self.children[move_nb] = MCTSNode(puzzle=child,
                                              parent=self,
                                              action=move_nb,
                                              heuristic=self.heuristic,
                                              c=self.c,
                                              nu=self.nu)
        scaling = sum(self.p.values())
        self.p = {action: proba/scaling for action, proba in self.p.items()}
        if self.parent is not None:
            self.parent.backward_prop(self.action, min(self.w.values()))

    def backward_prop(self, action, w):
        if w < self.w[action]:
            self.w[action] = w
            self.log_info('updating w[', action, '] -> ', w)
            if self.parent is not None:
                self.parent.backward_prop(self.action, min(self.w.values()))

    def choose_next(self):
        best_value = -inf
        best_action = None
        num = sqrt(sum(self.n.values()))
        debug_actions = dict()
        for action in self.p.keys():
            u = self.c * self.p[action] * num / (1 + self.n[action])
            q = self.w[action] + self.l[action]
            if u - q > best_value:
                best_value = u - q
                best_action = action
            debug_actions[action] = u - q
        self.log_info('choose_next -> ', best_action, ' debug_actions = ', debug_actions)
        self.n[action] += 1
        self.l[action] += self.nu
        return self.children[best_action]

    def construct_solution(self) -> Solution:
        cost = 0
        node = self
        path = list()
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
                        success=True)

########################################################################################################################


class MonteCarloTreeSearchSolver(Solver):
    """ We implement section 4.2 of the Agostinelli paper which we saved at
     rubiks.papers.SolvingTheRubiksCubeWithoutHumanKnowledge.pdf """

    c = 'c'
    nu = 'nu'

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

    def __init__(self, **kw_args):
        Solver.__init__(self, **kw_args)
        self.run_time = None
        self.tree = None
        self.expanded_nodes = 0
        self.puzzle = None
        self.heuristic = Heuristic.factory(**self.get_config())
        assert self.heuristic.heuristic_type == Heuristic.deep_q_learning, \
            'MonteCarloTreeSearchSolver only works with %s heuristic' % Heuristic.deep_q_learning

    def known_to_be_optimal(self):
        return False

    def solve_impl(self, puzzle, **kw_args):
        self.run_time = -snap()
        self.puzzle = puzzle
        self.log_info('Solving ', puzzle)
        while True:
            try:
                if self.run_time + snap() > self.time_out:
                    raise TimeoutError()
                self.log_info('go_to_leaf')
                leaf = self.go_to_leaf()
                self.log_info(' leaf = ', leaf)
                if leaf.is_goal():
                    self.log_info(' done ... will construct solution and return')
                    solution = leaf.construct_solution()
                    solution.expanded_nodes = self.expanded_nodes
                    solution.set_additional_info(solver_name=self.get_name())
                    return solution
                self.log_info(' not done')
            except TimeoutError:
                solution = Solution.failure(puzzle,
                                            time_out=True,
                                            failure_reson='time out')
                solution.expanded_nodes = self.expanded_nodes
                return solution

    def go_to_leaf(self) -> MCTSNode:
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
        node.expand()
        self.expanded_nodes = 1
        return node.choose_next()

########################################################################################################################
