########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from math import inf
from torch.nn import MSELoss, CrossEntropyLoss
########################################################################################################################
from rubiks.learners.deepreinforcementlearner import DeepReinforcementLearner
from rubiks.deeplearning.deeplearning import DeepLearning
########################################################################################################################


class DeepQLearner(DeepReinforcementLearner):
    """ This learner will learn jointly a policy and value function via deep reinforcement learning
    We follow simply the Agostinelli paper saved at rubiks.papers.SolvingTheRubiksCubeWithoutHumanKnowledge.pdf
    There are few modifications necessary versus my DeepReinforcementLearner. Only the target to be
    constructed need to return an action in addition to the value function, and the loss function is different.
    """

    def __init__(self, **kw_args):
        kw_args[DeepLearning.joint_policy] = True
        DeepReinforcementLearner.__init__(self, **kw_args)

    @classmethod
    def get_loss_function(cls):
        cross_entropy_loss = CrossEntropyLoss()
        mse_loss = MSELoss()

        def loss(output, target):
            return mse_loss(output[:, 0], target[:, 0]) + \
                   cross_entropy_loss(output[:, 1:], target[:, 1].long())
        return loss

    def __construct_target__(self, known_values, puzzle):
        try:
            action = 0
            target = 0
            if not puzzle.is_goal():
                target = inf
                possible = puzzle.possible_moves()
                for move_nb, move in enumerate(puzzle.theoretical_moves()):
                    if move not in possible:
                        continue
                    one_away_puzzle = puzzle.apply(move)
                    candidate_target = self.target_network.evaluate(one_away_puzzle)
                    candidate_target = candidate_target[0].item()
                    one_away_puzzle_hash = hash(one_away_puzzle)
                    if one_away_puzzle_hash in known_values:
                        candidate_target = min(candidate_target, known_values[one_away_puzzle_hash][0])
                    candidate_target += move.cost()
                    if candidate_target < target:
                        target = max(0, min(target, candidate_target))
                        action = move_nb
        except KeyboardInterrupt:
            return None
        if self.cap_target_at_network_count:
            target = min(target, self.target_network_count)
        return target, action

    @staticmethod
    def get_min_target(targets):
        return min(targets[:, 0]).item()

    @staticmethod
    def get_max_target(targets):
        return max(targets[:, 0]).item()

########################################################################################################################
