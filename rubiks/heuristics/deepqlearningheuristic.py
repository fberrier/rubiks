########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from torch.nn import Softmax
########################################################################################################################
from rubiks.heuristics.deeplearningheuristic import DeepLearningHeuristic
########################################################################################################################


class DeepQLearningHeuristic(DeepLearningHeuristic):
    """ similar to DeepLearningHeuristic but also can tell us something about
    actions we should take, in addition to cost-to-go """

    def __init__(self, model_file_name, **kw_args):
        DeepLearningHeuristic.__init__(self, model_file_name, **kw_args)
        self.soft_max = Softmax(dim=0)

    def cost_to_go_from_puzzle_impl(self, puzzle):
        self.check_puzzle(puzzle)
        return self.deep_learning.evaluate(puzzle)[0].item()

    def optimal_actions(self, puzzle):
        self.check_puzzle(puzzle)
        return {action: proba for action, proba in zip(puzzle.theoretical_moves(),
                                                       self.soft_max(self.deep_learning.evaluate(puzzle)[1:]).tolist())}

########################################################################################################################
