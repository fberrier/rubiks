########################################################################################################################
# Francois Berrier - Royal Holloway University London - MSc Project 2022                                               #
########################################################################################################################
from numpy import prod
from torch import stack
from torch.nn import Sequential, ReLU, BatchNorm1d, Linear
########################################################################################################################
from rubiks.puzzle.puzzle import Puzzle
from rubiks.deeplearning.deeplearning import DeepLearning
########################################################################################################################


class FullyConnected(DeepLearning):
    """ TBD """

    network_type = DeepLearning.fully_connected_net

    def __init__(self, puzzle_type, **kw_args):
        DeepLearning.__init__(self, puzzle_type, **kw_args)
        in_channels = prod(self.puzzle_dimension) ** 2
        self.layers = Sequential(Linear(in_channels, 1000),
                                 ReLU(),
        #                         BatchNorm1d(1000),
                                 Linear(1000, 500),
                                 ReLU(),
        #                         BatchNorm1d(500),
                                 Linear(500, 100),
                                 ReLU(),
        #                         BatchNorm1d(100),
                                 Linear(100, 1))

    def name(self):
        name = self.__class__.__name__
        if hasattr(self, 'layers'):
            name += '[%s]' % self.layers
        return name

    def evaluate(self, puzzles):
        if isinstance(puzzles, Puzzle):
            x = puzzles.to_tensor().float().reshape(-1).unsqueeze(0)
        elif isinstance(puzzles, list):
            x = stack([puzzle.to_tensor().float().reshape(-1) for puzzle in puzzles])
        else:
            raise NotImplementedError
        return self.forward(x)

    def forward(self, x):
        return self.layers(x).squeeze()
    
########################################################################################################################
