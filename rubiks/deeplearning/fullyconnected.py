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

    def __init__(self, puzzle_dimension, **kw_args):
        DeepLearning.__init__(self, puzzle_dimension, **kw_args)
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

    def evaluate(self, puzzles):
        if isinstance(puzzles, Puzzle):
            x = puzzles.to_tensor().float().reshape(-1).unsqueeze(0)
        elif isinstance(puzzles, list):
            x = stack([puzzle.to_tensor().float().reshape(-1) for puzzle in puzzles])
        else:
            raise NotImplementedError
        return self.forward(x)

    def forward(self, x):
        #self.log_info('x.shape: ', x.shape)
        #self.log_info('x: ', x)
        #self.log_info('x.dtype: ', x.dtype)
        y = self.layers(x).squeeze()
        #self.log_info('y.shape: ', y.shape)
        return y
    
########################################################################################################################
